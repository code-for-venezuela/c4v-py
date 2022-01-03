"""
    This file implements a class that manages local storage 
    using regular files.
"""
# Python
import sqlite3
from typing import Any, Iterator, Dict, List, Tuple
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

DATE_FORMAT = settings.date_format


class BigQueryManager(BasePersistencyManager):
    """
        Use Big Query based storage 
        to manage persistency.

        # Parameters 
        - table_name : `str` = name of the big query table (prefixed with dataset name) where to perform operations

    """

    def __init__(self, table_name : str, bq_client : bigquery.Client):
        self._table_name = table_name
        self._client = bq_client

    @property
    def table_name(self) -> str:
        return self._table_name

    @property
    def client(self) -> bigquery.Client:
        return self._client

    def get_all(
        self, limit: int = -1, scraped: bool = None, order_by: List[str] = None
    ) -> Iterator[ScrapedData]:

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

    def filter_scraped_urls(self, urls: List[str]) -> List[str]:

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

    def filter_known_urls(self, urls: List[str]) -> List[str]:

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

    def was_scraped(self, url: str) -> bool:

        # Connect to db and check if object was scraped

        query_str = f"SELECT 1 FROM `{self.table_name}` WHERE last_scraped IS NOT NULL AND url=@url;"
        job_config = bigquery.QueryJobConfig(query_parameters=[
            bigquery.ScalarQueryParameter("url", "STRING", url)
        ])

        obj = [_ for _ in self.client.query(query_str, job_config=job_config)]
        
        return obj != []

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
        self, order_by: str
    ) -> Tuple[str, str]:
        """
            Parse order and name of field from the given str
            Parameters:
                + order_by : str = A string with the following format:
                    (+|-)<field_name> where field_name is the name of a field in ScrapedData,
                    '+' corresponds to a an ascending order, and '-' corresponds to a descending order
            Return:
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
