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
            - `crawler_names` : [str] = list of crawlers to use while crawling
            - `limit` : int = (optional) Max ammount of urls to save, defaults to 1000
        # Environment Variables
            These are the expected environment variables that this function will try to use
            - `TABLE` : str = Name of the table in big query to use to store results
    """

    # Get logging objects
    logging_client = logging.Client()
    logger = logging_client.logger("microscope-crawl")

    # Parse config 
    config = CrawlFuncConfig(request)
    if config.error:
        logger.log_text(config.error, severity="ERROR")
        print(config.error)
        return flask.jsonify({"status":"error", "msg" : config.error})

    # Crawl new urls
    logger.log_text(f"Crawling up to {config.limit} urls using crawler '{config.crawler_names}'")
    crawled = config.manager.crawl_new_urls_for(config.crawler_names, config.limit)
    logger.log_text(f"Crawled {len(crawled)} urls using crawler '{config.crawler_names}'")
    
    return flask.jsonify({"status" : "success", "crawled" : len(crawled), "crawler_names" : config.crawler_names})

class CrawlFuncConfig:
    """
        This class holds the set up for the crawl function
    """
    DEFAULT_LIMIT : int = 100

    def __init__(self, request : flask.Request) -> None:

        # Parse options from request
        request_json : Dict = request.get_json() or {}

         # Parse table name fron environment variables
        table_name = os.environ.get("TABLE")
        if not table_name:
            self._error = "No table name provided in environment variables"
            return

        # Parse project id for firestore
        project_id = os.environ.get("PROJECT_ID")
        if not project_id:
            self._error = "No PROJECT_ID  provided in environment variables"
            return

        # Parse limit
        try:
            self._limit = int(request_json.get("limit", self.DEFAULT_LIMIT))
        except ValueError:
            self._error = "'limit' field in request should be a valid integer value"
            return 

        # Parse crawler name 
        self._ks = request_json.get("crawler_names")
        if not self.crawler_names:
            self._error = "Mandatory field 'crawler_names' not provided in request"
            return

        # Check that crawler name is a valid crawler name
        for crawler_name in self.crawler_names:
            if not crawler_name in Manager.available_crawlers():
                self._error = f"Crawler '{crawler_name}' not a valid crawler, available crawlers: {Manager.available_crawlers()}"
                return

        # set up driver data
        self._scrape_data_table_name = table_name
        client = CrawlFuncConfig.get_client()
        self._manager = Manager(BigQueryManager(table_name, client, project_id))

        self._error = None

    @property
    def limit(self) -> int:
        """
            Max ammount of urls to crawl, defaults to 100
        """
        return self._limit

    @property
    def crawler_names(self) -> str:
        """
            Name of the crawler to be run
        """
        return self._crawler_names

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
