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
from c4v.scraper.scraped_data_classes.scraped_data import ServiceClassificationLabels, RelevanceClassificationLabels, ScrapedData, Sources
from c4v.config import settings
import uuid

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
                                (you can fill it with the C4V_SCRAPED_DATA_TABLE env variable or setting variable)
        - bq_client : `bigquery.Client` = client object to use 
        - project_id : `str` = name of the project in gcloud to connect to in order to access firestore
                                (you can fill it by using the C4V_GCLOUD_PROJECT_ID variable)
    """                         

    _firestore_db = None

    def __init__(self, table_name : str, bq_client : bigquery.Client, project_num : str, gcloud_max_content_len : int = settings.gcloud_max_content_len):
        self._table_name = table_name
        self._client = bq_client
        self._project_num = project_num
        self._gcloud_max_content_len = gcloud_max_content_len
        self._init_firestore()

    @property
    def table_name(self) -> str:
        return self._table_name

    @property
    def client(self) -> bigquery.Client:
        return self._client

    @property
    def project_num(self) -> str:
        return self._project_num

    @property
    def _firestore_scraped_data_table_name(self) -> str:
        return u"scraped_data"

    @property 
    def _firestore_db_ref(self) -> Any:
        return self._firestore_db.collection(self._firestore_scraped_data_table_name)

    @property
    def bq_date_format(self) -> str:
        # Required to send data to big query on list of json 
        return  "%Y-%m-%d %H:%M:%S.%f"
    
    @property
    def gcloud_max_content_len(self) -> int:
        """
            Max size for body len for an article
        """
        return self._gcloud_max_content_len

    def _init_firestore(self):
        """
            Initialization code for firestore
        """
        if (BigQueryManager._firestore_db != None):
            return # nothing to init

        self._credentials = credentials.ApplicationDefault()
        firebase_admin.initialize_app(self._credentials, {
        # 'projectId': self._project_num,
        })
        BigQueryManager._firestore_db = firestore.client()


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
            query += f" LIMIT {limit};"

        res = self.client.query(query)
        for row in res:
            # Decompose row
            (url, last_scraped, title, content, author, date, label_relevance, label_service, source, categories) = row

            if label_relevance:
                try:
                    label_relevance = RelevanceClassificationLabels(label_relevance)
                except:  # not a known label
                    label_relevance = RelevanceClassificationLabels.UNKNOWN
            else:
                label_relevance = None

            if label_service:
                try:
                    label_service = ServiceClassificationLabels(label_service)
                except:  # not a known label
                    label_service = ServiceClassificationLabels.UNKNOWN
            else:
                label_service = None

            if source:
                try:
                    source = Sources(source)
                except:  # unknown source
                    source = Sources.UNKOWN

            yield ScrapedData(
                url=url,
                last_scraped=last_scraped,
                title=title,
                content=content,
                author=author,
                date=date,
                categories=categories,
                label_relevance=label_relevance,
                label_service=label_service,
                source=source,
            )
        
    def _get_all_from_firestore(
        self, limit: int = -1, scraped: bool = None, order_by: List[str] = None, ready : bool = None
        ) -> Iterator[ScrapedData]:
        """
            Get all data from Firestore according to the get all arguments.
            Firestore data will have the following format:
                <scraped data fields>
                - was_scraped : bool = if this row is already scraped
                - is_ready_for_bq : bool = if this row is ready to be upgraded to bq
            
            Use the "ready" argument to tell if you want the retrieved data to be ready or if you just want the 
            rows not yet ready or already ready
        """
        # Sanity check
        assert isinstance(limit, int), "limit should be integer"

        # If nothing to retrieve, just finish
        if limit == 0:
            return

        scraped_data_ref = self._firestore_db_ref

        # If scraped specified
        if scraped == True:
            scraped_data_ref = scraped_data_ref.where(u'was_scraped', u'==', True)
        elif scraped == False:
            scraped_data_ref = scraped_data_ref.where(u'was_scraped', u'==', False)
        
        # If ready specified
        if ready != None:
            assert isinstance(ready, bool), "ready argument should be bool"
            scraped_data_ref = scraped_data_ref.where(u'is_ready_for_bq', u'==', ready)

        # if order by is provided:
        if order_by:
            parsed_orders = (self._parse_order_and_field_from_order_by_str(order, True) for order in order_by)
            for (order, field) in parsed_orders:
                scraped_data_ref = scraped_data_ref.order_by(field, direction=order)

        # if limit is provided
        if limit > 0:
            scraped_data_ref = scraped_data_ref.limit(limit)

        return ( self._from_firestore_to_scraped_data(x) for x in scraped_data_ref.stream())
    
    def get_all(self, limit: int = -1, scraped: bool = None, order_by: List[str] = None) -> Iterator[ScrapedData]:
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


    def _from_firestore_to_scraped_data(self, data : Any) -> ScrapedData:
        """
            Used to convert from firestore stored object to actual scraped data object.
            It will remove the firestore-specific fields.
            'data' is an instance object as it comes from firestore
            
        """
        data = data.to_dict()
        del data['was_scraped']
        del data['is_ready_for_bq']

        return ScrapedData.from_dict(data)

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
        return self._filter_known_urls_bq(urls)

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
        # connect to db and check for each url if such url exists, if so, then it's known and scraped.
        # Otherwise it is unkown for the DB, and thus it's also not scraped
        query_str = f"SELECT url FROM `{self.table_name}` WHERE url in UNNEST(@url);"
        job_config = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ArrayQueryParameter("url", "STRING", urls)
            ]
        )

        # Create a set to discard results in such set 
        res : Set(str)= set(urls)

        # if last_scraped is null, then it wasn't scraped
        query = self.client.query(query_str, job_config=job_config)
        for row in query:
            (url,) = row
            # Remove urls in DB
            if url in res:
                res.remove(url)
        
        return list(res)

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

    def _save_bq(self, url_data: List[ScrapedData]):
        # Save data into database
        if not url_data:
            return

        data_to_insert = [dataclasses.asdict(data) for data in url_data]
        for data in data_to_insert:
            label_relevance: RelevanceClassificationLabels = data["label_relevance"]
            data["label_relevance"] = label_relevance.value if label_relevance else label_relevance 
            label_service: ServiceClassificationLabels = data["label_service"]
            data["label_service"] = label_service.value if label_service else label_service 
            source: Sources = data["source"]
            data["source"] = source.value if source else source
            data["last_scraped"] = datetime.datetime.strftime( data["last_scraped"], self.bq_date_format) if data["last_scraped"] else None

            # truncate content if necessary
            content : str = data.get("content")
            if content and len(content) > self.gcloud_max_content_len:
                data['content'] = content[:self.gcloud_max_content_len]
            

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

    def _save_firestore(self, url_data: List[ScrapedData]):
        # Perform save operation for firestore

        db_ref = self._firestore_db_ref

        # Convert into firestore-specific format
        fs_data = ((x.url , self._from_scraped_data_to_firestore(x)) for x in url_data)
        for (url, data) in fs_data:
            db_ref.document(self._get_url_uuid(url)).set(data)

    def save(self, url_data: List[ScrapedData]):
        # Note that a regular save operation won't save to BQ, it will go to firestore first
        return self._save_firestore(url_data)

    def _retrieve_and_delete_ready_from_firestore(self) -> List[ScrapedData]:
        """
            This function will try to retrieve all ready rows in DB, and then delete them
            in a transactional manner
        """
        # TODO maybe use transaction?
        data_ref = self._firestore_db_ref
        data_stream = data_ref.where("is_ready_for_bq", '==', True)

        results = []
        for data in data_stream.stream():
            # save each data row and then delete it
            sd = self._from_firestore_to_scraped_data(data)
            results.append(sd)
            data_ref.document(self._get_url_uuid(sd.url)).delete()
        
        return results

    def move(self):
        """
            Move ready data from firestore to big query, and then remove it from 
            firestore
        """
        data_to_move = self._retrieve_and_delete_ready_from_firestore()
        try:
            # try to save data to bq 
            self._save_bq(data_to_move)
        except Exception as e:
            # save it back to firestore if it wasn't possible
            self.save(data_to_move)
            

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

    @staticmethod
    def _get_url_uuid(url : str) -> str:
        """
            Use this function to get an uuid from an url, 
            useful to hash urls into ids in firestore
        """
        return str(uuid.uuid3(uuid.NAMESPACE_URL, url))

def _parse_dict_to_url_data(obj: Dict[str, Any]) -> ScrapedData:
    """
    Parse dict object into a new object instance
    """
    return ScrapedData(**obj)
