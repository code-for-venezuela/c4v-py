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
from c4v.scraper.scraped_data_classes.scraped_data import RelevanceClassificationLabels, ScrapedData, ServiceClassificationLabels, Sources
from c4v.config import settings

DATE_FORMAT = settings.date_format


class SqliteManager(BasePersistencyManager):
    """
        Use SQLite based local storage 
        to manage persistency.
    """

    def __init__(self, db_path: str):
        self._db_path = db_path
        self._migrate(db_path)

    def _migrate(self, db_path: str):
        """
            Perform a database migration, ensuring that tables are as specified in this class,
            use db specified by given path
        """
        with sqlite3.connect(db_path) as connection:
            cursor = connection.cursor()  # connect to cursor

            # All these migations are "if not exists" so no change is performed when they already exist
            self._create_scraped_data_table(cursor)  # create scraped data table
            self._create_category_table(
                cursor
            )  # category table (including many to many)
            connection.commit()  # save changes

    def _create_scraped_data_table(self, cursor: sqlite3.Cursor):
        """
            Create scraped data with the structure as in ScrapedData model, perform changes with 
            provided cursor
        """
        cursor.execute(
            "CREATE TABLE IF NOT EXISTS  scraped_data(\
            url PRIMARY KEY,\
            last_scraped DATETIME NULL,\
            title TEXT NULL,\
            content TEXT NULL,\
            author TEXT NULL,\
            date DATETIME NULL,\
            label_relevance TEXT NULL, \
            label_service TEXT NULL, \
            source TEXT NULL\
            );"
        )

    def _create_category_table(self, cursor: sqlite3.Cursor):
        """
            Create category table, including a many to many relationship between categories.
            Perform changes with provided cursor
        """
        cursor.execute(
            "CREATE TABLE IF NOT EXISTS category(\
            name PRIMARY KEY\
        );"
        )
        cursor.execute(
            "CREATE TABLE IF NOT EXISTS  category_to_data (\
                scraped_data_url,\
                category_name,\
                FOREIGN KEY (scraped_data_url) REFERENCES scraped_data(url) ON DELETE CASCADE,\
                FOREIGN KEY (category_name) REFERENCES category(name) ON DELETE CASCADE,\
                CONSTRAINT uniq UNIQUE (scraped_data_url, category_name)\
            );"
        )

    def get_all(
        self, limit: int = -1, scraped: bool = None, order_by: List[str] = None
    ) -> Iterator[ScrapedData]:

        # Retrieve all data stored
        with sqlite3.connect(self._db_path) as connection:
            cursor = connection.cursor()
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
                new_cur = cursor.execute(
                    f"SELECT * FROM scraped_data WHERE last_scraped IS NOT NULL{parsed_orders};",
                )
            elif scraped == False:
                new_cur = cursor.execute(
                    f"SELECT * FROM scraped_data WHERE last_scraped IS NULL{parsed_orders};",
                )
            else:
                new_cur = cursor.execute(f"SELECT * FROM scraped_data{parsed_orders};",)

            # If limit less than 0, then take as much as you can
            if limit < 0:
                res = new_cur.fetchall()
            else:
                res = new_cur.fetchmany(limit)

            for row in res:
                # Decompose row
                (url, last_scraped, title, content, author, date, label_relevance, label_service, source) = row

                if label_relevance:
                    try:
                        label_relevance = RelevanceClassificationLabels(label_relevance)
                    except:  # not a known label
                        label_relevance = RelevanceClassificationLabels.UNKNOWN

                if label_service:
                    try:
                        label_service = ServiceClassificationLabels(label_service)
                    except:  # not a known label
                        label_service = ServiceClassificationLabels.UNKNOWN

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

                categories = [
                    category
                    for (category,) in cursor.execute(
                        "SELECT category_name FROM category_to_data WHERE scraped_data_url=?;",
                        [url],
                    )
                ]

                yield ScrapedData(
                    url=url,
                    last_scraped=last_scraped,
                    title=title,
                    content=content,
                    author=author,
                    date=date,
                    categories=categories,
                    label_relevance=label_relevance,
                    source=source,
                )

    def filter_scraped_urls(self, urls: List[str]) -> List[str]:

        # connect to db and check for each url if such url was scraped,
        # checking its last_scrape field

        res = []
        with sqlite3.connect(self._db_path) as connection:

            cursor = connection.cursor()
            for url in urls:

                # if last_scraped is null, then it wasn't scraped
                is_there = cursor.execute(
                    "SELECT 1 FROM scraped_data WHERE url=? AND last_scraped IS NOT NULL",
                    [url],
                ).fetchone()

                if not is_there:
                    res.append(url)

        return res

    def filter_known_urls(self, urls: List[str]) -> List[str]:

        # connect to db and check for each url if such url was scraped,
        # checking its last_scrape field

        res = []
        with sqlite3.connect(self._db_path) as connection:

            cursor = connection.cursor()
            for url in urls:

                # if last_scraped is null, then it wasn't scraped
                is_there = cursor.execute(
                    "SELECT 1 FROM scraped_data WHERE url=?", [url],
                ).fetchone()

                if not is_there:
                    res.append(url)

        return res

    def was_scraped(self, url: str) -> bool:

        # Connect to db and check if object was scraped
        with sqlite3.connect(self._db_path) as connection:
            cursor = connection.cursor()

            obj = cursor.execute(
                "SELECT ? FROM scraped_data WHERE last_scraped IS NOT NULL", [url]
            ).fetchone()

            return obj != None

    def save(self, url_data: List[ScrapedData]):
        # Save data into database
        if not url_data:
            return

        with sqlite3.connect(self._db_path) as connection:
            cursor = connection.cursor()

            data_to_insert = [dataclasses.asdict(data) for data in url_data]
            for data in data_to_insert:
                label_relevance: RelevanceClassificationLabels = data["label_relevance"]
                data["label_relevance"] = label_relevance.value if label_relevance else label_relevance

                label_service: ServiceClassificationLabels = data['label_service']
                data["label_service"] = label_service.value if label_service else label_service

                source: Sources = data["source"]
                data["source"] = source.value if source else source

            cursor.executemany(
                "INSERT OR REPLACE INTO scraped_data VALUES (:url, :last_scraped, :title, :content, :author, :date, :label_relevance, :label_service, :source)",
                data_to_insert,
            )

            for data in url_data:
                # insert new categories
                # may have none if category field is set to None
                if data.categories is None:
                    continue

                cursor.executemany(
                    "INSERT OR IGNORE INTO category VALUES (?)",
                    [(cat,) for cat in data.categories],
                )

                # insert many to many relationship
                cursor.executemany(
                    "INSERT OR IGNORE INTO category_to_data VALUES (?, ?)",
                    [(data.url, category) for category in data.categories],
                )

            # save changes
            connection.commit()

    def delete(self, urls: List[str]):

        # Bulk delete provided urls
        with sqlite3.connect(self._db_path) as connection:
            cursor = connection.cursor()
            cursor.executemany(
                "DELETE FROM scraped_data WHERE url=?", [(url,) for url in urls]
            )
            connection.commit()

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
