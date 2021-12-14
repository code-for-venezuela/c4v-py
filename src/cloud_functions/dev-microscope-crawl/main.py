# Python imports
from typing import Dict
import flask
import os

# Local imports
from c4v.microscope import Manager
from c4v.scraper.persistency_manager.big_query_persistency_manager import BigQueryManager

# Third Party Imports
from google.cloud import bigquery, logging


def crawl(request : flask.Request):
    """
        function to trigger a crawling process in google cloud. 
        this function expects the following parameters through post request:
        # Parameters:
            - `crawler_name` : str = name of the crawler to use while crawling
            - `limit` : int = (optional) Max ammount of urls to save, defaults to 1000
        # Environment Variables
            These are the expected environment variables that this function will try to use
            - `TABLE` : str = Name of the table in big query to use to store results
    """
    # Get logging objects
    logging_client = logging.Client()
    logger = logging_client.logger("microscope-crawl")

    # Parse table name fron environment variables
    table_name = os.environ.get("TABLE")
    if not table_name:
        logger.log_text("Not table name provided in environment variables", severity = "ERROR")
        return "error"
    
    # Parse config 
    config = CrawlFuncConfig(table_name, request)
    if config.error:
        logger.log_text(config.error, severity="ERROR")
        return "error"

    # Crawl new urls
    logger.log_text(f"Crawling up to {config.limit} urls using crawler '{config.crawler_name}'")
    crawled = config.manager.crawl_new_urls_for([config.crawler_name], config.limit)
    logger.log_text(f"Crawled {len(crawled)} urls using crawler '{config.crawler_name}'")
    
    return "success"

class CrawlFuncConfig:
    """
        This class holds the set up for the crawl function
    """
    DEFAULT_LIMIT : int = 100

    def __init__(self, scraped_data_table_name : str, request : flask.Request) -> None:

        # Parse options from request
        request_json : Dict = request.get_json()

        # Parse limit
        try:
            self._limit = int(request_json.get("limit", self.DEFAULT_LIMIT))
        except ValueError:
            self._error = "'limit' field in request should be a valid integer value"
            return 

        # Parse crawler name 
        self._crawler_name = request_json.get("crawler_name")
        if not self._crawler_name:
            self._error = "Mandatory field 'crawler_name' not provided in request"
            return

        # Check that crawler name is a valid crawler name
        if not self.crawler_name in Manager.available_crawlers():
            self._error = f"Crawler '{self.crawler_name}' not a valid crawler, available crawlers: {Manager.available_crawlers()}"
            return

        # set up driver data
        self._scrape_data_table_name = scraped_data_table_name
        client = CrawlFuncConfig.get_client()
        self._manager = Manager(BigQueryManager(scraped_data_table_name, client))

        self._error = None

    @property
    def limit(self) -> int:
        """
            Max ammount of urls to crawl, defaults to 100
        """
        self._limit

    @property
    def crawler_name(self) -> str:
        """
            Name of the crawler to be run
        """
        self._crawler_name

    @property
    def manager(self) -> Manager:
        return self._manager

    @property
    def scrape_data_table_name(self) -> str:
        return self._scrape_data_table_name

    @property
    def error(self) -> str:
        return self._error

    @staticmethod
    def get_client() -> bigquery.Client:
        client = bigquery.Client()
        return client 
