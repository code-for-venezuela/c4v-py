"""
    This file implements a class that manages local storage 
    using regular files.
"""
# Python
from typing import Any, Iterator, Dict, List, Tuple, Set
import dataclasses
import datetime

# Local imports
from c4v.scraper.persistency_manager.base_persistency_manager import (
    BasePersistencyManager,
)
from c4v.scraper.scraped_data_classes.scraped_data import Labels, ScrapedData, Sources
from c4v.config import settings

# Third party imports
from google.cloud import bigquery
import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore
from c4v.scraper.scraped_data_classes.scraped_data import ScrapedData
import datetime
import pytz



DATE_FORMAT = settings.date_format


class BigQueryManager(BasePersistencyManager):
    """
        Use Big Query based storage 
        to manage persistency.

        Recently added and fill-pending measurements are stored in firestore, 
        while complete data is stored in big query, as it is more reliable 
        for massive ammounts of data.

        Big query is read and insert only for this manager object.

        # Parameters 
        - table_name : `str` = name of the big query table (prefixed with dataset name) where to perform operations
        - project_id : `str` = name of the project in gcloud to connect to in order to access firestore
    """

    def __init__(self, table_name : str, bq_client : bigquery.Client, project_id : str):
        self._table_name = table_name
        self._client = bq_client
        self._project_id = project_id
        self._init_firestore()

    @property
    def table_name(self) -> str:
        return self._table_name

    @property
    def client(self) -> bigquery.Client:
        return self._client

    @property
    def project_id(self) -> str:
        return self._project_id

    @property
    def _firestore_scraped_data_table_name(self) -> str:
        return u"scraped_data"

    def _init_firestore(self):
        """
            Initialization code for firestore
        """
        self._credentials = credentials.ApplicationDefault()
        firebase_admin.initialize_app(self._credentials, {
        'projectId': "event-pipeline",
        })
        self._firestore_db = firestore.client()


    def _get_all_from_bq(
        self, limit: int = -1, scraped: bool = None, order_by: List[str] = None
    ) -> Iterator[ScrapedData]:
        """
            Get all data from Big Query according to the get all arguments
        """
        # Retrieve all data stored
        
        # Create the query string
        # Order
        if order_by:
            parsed_orders = " ORDER BY " + ", ".join(
                [
                    f"{name} {order}"
                    for (order, name) in (
                        self._parse_order_and_field_from_order_by_str(order_by=ord)
                        for ord in order_by
                    )
                ]
            )
        else:
            parsed_orders = ""

        # if scraped = true, take only scraped. If false, take non-scraped. If none, take all
        if scraped:
            query = f"SELECT * FROM `{self.table_name}` WHERE last_scraped IS NOT NULL{parsed_orders}"
        elif scraped == False:
            query = f"SELECT * FROM `{self.table_name}` WHERE last_scraped IS NULL{parsed_orders}"
        else:
            query = f"SELECT * FROM `{self.table_name}` {parsed_orders}"

        # If limit less than 0, then take as much as you can
        assert isinstance(limit, int), "limit argument should be a valid integer"
        if limit < 0:
            query += ";"
        else:
            query += f" LIMIT={limit};"

        res = self.client.query(query)
        for row in res:
            # Decompose row
            (url, last_scraped, title, content, author, date, label_relevance, label_service, source, categories) = row

            if label_relevance:
                try:
                    label = Labels(label_relevance)
                except:  # not a known label
                    label = Labels.UNKNOWN
            else:
                label = None

            if source:
                try:
                    source = Sources(source)
                except:  # unknown source
                    source = Sources.UNKOWN

            # parse date to datetime:
            try:
                last_scraped = (
                    datetime.datetime.strptime(last_scraped, DATE_FORMAT)
                    if last_scraped
                    else last_scraped
                )
            except ValueError as _:  # In case it fails using a format not valid for python3.6
                for i in range(len(last_scraped) - 1, -1, -1):
                    if last_scraped[i] == ":":
                        last_scraped = last_scraped[:i] + last_scraped[i + 1 :]
                        break
                last_scraped = (
                    datetime.datetime.strptime(last_scraped, DATE_FORMAT)
                    if last_scraped
                    else last_scraped
                )

            yield ScrapedData(
                url=url,
                last_scraped=last_scraped,
                title=title,
                content=content,
                author=author,
                date=date,
                categories=categories,
                label=label,
                source=source,
            )
        
    def _get_all_from_firestore(
        self, limit: int = -1, scraped: bool = None, order_by: List[str] = None
        ) -> Iterator[ScrapedData]:
        """
            Get all data from Firestore according to the get all arguments.
            Firestore data will have the following format:
        """
        # Sanity check
        assert isinstance(limit, int), "limit should be integer"

        # If nothing to retrieve, just finish
        if limit == 0:
            return

        scraped_data_ref = self._firestore_db.collection(self._firestore_scraped_data_table_name)

        # If scraped specified
        if scraped == True:
            scraped_data_ref = scraped_data_ref.where(u'was_scraped', u'==', True)
        elif scraped == False:
            scraped_data_ref = scraped_data_ref.where(u'was_scraped', u'==', False)
        
        # if order by is provided:
        if order_by:
            parsed_orders = (self._parse_order_and_field_from_order_by_str(order, True) for order in order_by)
            for (order, field) in parsed_orders:
                scraped_data_ref = scraped_data_ref.order_by(field, direction=order)

        # if limit is provided
        if limit > 0:
            scraped_data_ref = scraped_data_ref.limit(limit)

        return ( self._from_firestore_to_scraped_data(x.to_dict()) for x in scraped_data_ref.stream())
    
    def get_all(self, limit: int, scraped: bool, order_by: List[str] = None) -> Iterator[ScrapedData]:
        # Merge data from firestore to bq
        # TODO Create a merge function to merge results in both queries

        returned = 0
        for x in self._get_all_from_firestore(limit, scraped , order_by):
            yield x
            returned += 1

        # Sanity check
        assert returned <= limit or limit < 0, "There should not be more returned urls than the provided limit"

        # Check if already at the max 
        if returned == limit and limit > 0:
            return

        # This is an optimization, there's no non-scraped data in BQ
        if scraped == False: 
            return 

        for x in self._get_all_from_bq(limit, scraped , order_by):
            yield x
            

    def _from_firestore_to_scraped_data(self, dict_data : Dict[str, Any]) -> ScrapedData:
        """
            Used to convert from firestore stored object to actual scraped data object.
            It will remove the firestore-specific fields
        """
        del dict_data['was_scraped']
        del dict_data['is_ready_for_bq']

        return ScrapedData.from_dict(dict_data)

    def _from_scraped_data_to_firestore(self, scraped_data : ScrapedData) -> ScrapedData:
        """
            Used to convert from actual scraped data to an actual firestore storable object.
            It will add firestore-specific fields:
                - was_scraped : bool = tells if this instance is already scraped
                - is_ready_for_bq : bool = tells if this instance can be saved into big query
        """
        d = scraped_data.to_dict()
        d['was_scraped'] = scraped_data.is_scraped
        d["is_ready_for_bq"] = scraped_data.label != None and scraped_data.is_scraped

        return d
    
    def _filter_scraped_urls_bq(self, urls: List[str]) -> List[str]:

        # connect to db and check for each url if such url was crawled,
        # checking its last_scrape field

        res = []

        query_str = f"SELECT 1 FROM `{self.table_name}` WHERE url=@url AND last_scraped IS NOT NULL;"
        for url in urls:
            job_config = bigquery.QueryJobConfig(
                query_parameters=[
                    bigquery.ScalarQueryParameter("url", "STRING", url),
                ]
            )

            # if last_scraped is null, then it wasn't scraped
            is_there = self.client.query(query_str, job_config=job_config)

            if not is_there:
                res.append(url)

        return res

    def _filter_scraped_urls_firestore(self, urls: List[str]) -> List[str]:
        # Get table reference
        scraped_data_ref = self._firestore_db.collection(self._firestore_scraped_data_table_name)

        # Max ammount of elements in "in" query of firestore
        chunk_size = 10

        # get urls such that there's not 
        lo = 0
        result = []
        while (lo < len(urls)):
            # Current chunk to check
            chunk = set(urls[lo:lo + chunk_size])

            # Urls that are both in the list and scraped
            scraped_urls = {x.get("url") for x in scraped_data_ref.where("url", 'in', chunk).where("was_scraped", "==", True).stream()}
            
            for url in chunk:
                if url not in scraped_urls:
                    result.append(url)

            # firestore has a max "in" query size of 10
            lo += chunk_size
        
        return result

    def filter_scraped_urls(self, urls: List[str]) -> List[str]:
        # Just filter from firestore what was filtered from bq. 
        # Note that bq might be a more precise tool as it holds only complete data
        return self._filter_scraped_urls_firestore(self._filter_scraped_urls_bq(urls))

    def _filter_known_urls_bq(self, urls: List[str]) -> List[str]:

        # connect to db and check for each url if such url was scraped,
        # checking its last_scrape field

        res = []
        query_str = f"SELECT 1 FROM `{self.table_name}` WHERE url=@url;"
        for url in urls:
            # if last_scraped is null, then it wasn't scraped
            job_config = bigquery.QueryJobConfig(
                query_parameters= [
                    bigquery.ScalarQueryParameter("url", "STRING", url),
                ]
            )

            is_there = self.client.query(query_str, job_config=job_config)

            if not is_there:
                res.append(url)

        return res

    def _filter_known_urls_firestore(self, urls: List[str]) -> List[str]:
        # Get table reference
        scraped_data_ref = self._firestore_db.collection(self._firestore_scraped_data_table_name)

        # Max ammount of elements in "in" query of firestore
        chunk_size = 10

        # get urls such that there's not 
        lo = 0
        result = []
        while (lo < len(urls)):
            # Current chunk to check
            chunk = urls[lo:lo + chunk_size]

            # Urls that are both in the list and scraped
            known_urls = {x.get("url") for x in scraped_data_ref.where("url", 'in', chunk).stream()}
            
            for url in chunk:
                if url not in known_urls:
                    result.append(url)

            # firestore has a max "in" query size of 10
            lo += chunk_size
        
        return result

    def filter_known_urls(self, urls: List[str]) -> List[str]:
        # return urls that are neither in bq nor in firestore
        return self._filter_known_urls_firestore(self._filter_known_urls_bq(urls))

    def _was_scraped_bq(self, url: str) -> bool:
        # Connect to db and check if object was scraped
        query_str = f"SELECT 1 FROM `{self.table_name}` WHERE url=@url;"
        job_config = bigquery.QueryJobConfig(query_parameters=[
            bigquery.ScalarQueryParameter("url", "STRING", url)
        ])

        for _ in self.client.query(query_str, job_config=job_config):
            return True # consume just one, altought there should be at the most one element in this query
        
        return False

    def _was_scraped_firestore(self, url: str) -> bool:
        # Get reference to table
        scraped_data_ref = self._firestore_db.collection(self._firestore_scraped_data_table_name)

        # check if exists
        for _ in scraped_data_ref.where('url', '==', url).where('was_scraped', '==', True).stream():
            return True
        
        return False

    def was_scraped(self, url: str) -> bool:
        # Just check if it was scraped in some database
        return self._was_scraped_bq(url) or self._was_scraped_firestore(url)

    def save(self, url_data: List[ScrapedData]):
        # Save data into database
        if not url_data:
            return

        data_to_insert = [dataclasses.asdict(data) for data in url_data]
        for data in data_to_insert:
            ##### Should be gone in the future:
            label: Labels = data["label"]
            del data['label']
            data["label_relevance"] = label.value if label else label 
            data["label_service"]   = None 
            ###################################
            source: Sources = data["source"]
            data["source"] = source.value if source else source
            data["last_scraped"] = datetime.datetime.strftime( data["last_scraped"], settings.date_format) if data["last_scraped"] else None
        del url_data

        # Delete instances before insert them to give it priority to the ones to be saved right now
        query_str = f"DELETE FROM `{self.table_name}` WHERE url IN @urls;"
        query_config = bigquery.QueryJobConfig(query_parameters=
            [
                bigquery.ArrayQueryParameter("urls", "STRING", [d['url'] for d in data_to_insert])
            ]
        )
        query_job = self.client.query(query_str, job_config=query_config)

        # Now insert updated rows
        errors = self.client.insert_rows_json(self.table_name, data_to_insert)
        assert not errors, f"There was some errors while trying to insert new data to big query: {errors}"

    def delete(self, urls: List[str]):

        # Bulk delete provided urls
        query_str = f"DELETE FROM `{self.table_name}` WHERE url=@url;"
        for url in urls:
            job_config = bigquery.QueryJobConfig(query_parameters=
                [
                    bigquery.ScalarQueryParameter("url", "STRING", url)
                ]
            )
            self.client.query(query_str, job_config=job_config)

    def _parse_order_and_field_from_order_by_str(
        self, order_by: str, use_firestore_order : bool = False
    ) -> Tuple[str, str]:
        """
            Parse order and name of field from the given str
            # Parameters:
                - order_by : str = A string with the following format:
                    (+|-)<field_name> where field_name is the name of a field in ScrapedData,
                    '+' corresponds to a an ascending order, and '-' corresponds to a descending order
                - use_firestore_order : bool = if should use firestore order literals
            # Return:
                Tuple such that
                ('ASC' | 'DESC', <field_name>) 
        """

        # Check valid format
        if not order_by:
            raise ValueError(
                "Invalid value for order_by string, I'm getting an empty string"
            )
        elif order_by[0] != "-" and order_by[0] != "+":
            raise ValueError(
                f"Invalid value for order_by string, first char should be the order, on of: [-, +]. Provided string: {order_by}"
            )
        elif len(order_by) < 2:
            raise ValueError(
                f"Invalid value for order_by string, name of field not provided. Given string: {order_by}"
            )

        # Parsing order
        if use_firestore_order:
            order = firestore.Query.ASCENDING if order_by[0] == "+" else firestore.Query.DESCENDING
        else:
            order = "ASC" if order_by[0] == "+" else "DESC"

        # Parsing field name
        valid_fields = [d.name for d in dataclasses.fields(ScrapedData)]
        field_name = order_by[1:]

        # Raise error if not a valid string
        if field_name not in valid_fields:
            valid_fields_str = "\n".join([f"\t* {s}\n" for s in valid_fields])
            raise ValueError(
                f"Invalid value for order_by string, provided name doesn't match any field in ScrapedData. Provided field: {field_name}.\nAvailable fields:\n{valid_fields_str}"
            )

        return (order, field_name)

def _parse_dict_to_url_data(obj: Dict[str, Any]) -> ScrapedData:
    """
    Parse dict object into a new object instance
    """
    return ScrapedData(**obj)
