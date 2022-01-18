# Python imports
from typing import Dict
import flask
import os

# Local imports
from c4v.microscope import Manager
from c4v.scraper.persistency_manager.big_query_persistency_manager import BigQueryManager

# Third Party Imports
from google.cloud import bigquery, logging


def scrape(request : flask.Request):
    """
        function to trigger a scraping process in google cloud. 
        this function expects the following parameters through post request:
        # Parameters:
            - `limit` : int = (optional) Max ammount of urls to save, defaults to 1000
        # Environment Variables
            These are the expected environment variables that this function will try to use
            - `TABLE` : str = Name of the table in big query to use to store results
            - `PROJECT_ID` : str = Name of the project to where to store firestore data
    """
    # Get logging objects
    logging_client = logging.Client()
    logger = logging_client.logger("microscope-scrape")
    
    # Parse config 
    config = ScrapeFuncConfig(request)
    if config.error:
        logger.log_text(config.error, severity="ERROR")
        return {"status" : "error", "msg" : config.error} 

    # Crawl new urls
    logger.log_text(f"Scraping up to {config.limit} urls.")
    config.manager.scrape_pending(limit=config.limit)
    
    return {"status" : "success"} 

class ScrapeFuncConfig:
    """
        This class holds the set up for the crawl function
    """
    DEFAULT_LIMIT : int = 100

    def __init__(self, request : flask.Request) -> None:

        # Parse table name fron environment variables
        table_name = os.environ.get("TABLE")
        if not table_name:
            self._error = "Not table name provided in environment variables"
            return

        # Parse project id for firestore
        project_id = os.environ.get("PROJECT_ID")
        if not project_id:
            self._error = "No PROJECT_ID  provided in environment variables"
            return

        # Parse options from request
        request_json : Dict = request.get_json() or {}

        # Parse limit
        try:
            self._limit = int(request_json.get("limit", self.DEFAULT_LIMIT))
        except ValueError:
            self._error = "'limit' field in request should be a valid integer value"
            return 

        # set up driver data
        self._scrape_data_table_name = table_name
        self._project_id = project_id
        client = ScrapeFuncConfig.get_client()
        self._manager = Manager(BigQueryManager(table_name, client, project_id))

        self._error = None

    @property
    def limit(self) -> int:
        """
            Max ammount of urls to crawl, defaults to 100
        """
        return self._limit

    @property
    def manager(self) -> Manager:
        return self._manager

    @property
    def scrape_data_table_name(self) -> str:
        return self._scrape_data_table_name

    @property
    def project_id(self) -> str:
        return self._project_id

    @property
    def error(self) -> str:
        return self._error

    @staticmethod
    def get_client() -> bigquery.Client:
        client = bigquery.Client()
        return client 
