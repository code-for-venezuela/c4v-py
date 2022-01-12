# Python imports
import os

# Local imports 
from c4v.microscope import Manager
from c4v.scraper.persistency_manager.big_query_persistency_manager import BigQueryManager

# Third party imports
import flask
from google.cloud import bigquery, logging

def classify(request : flask.Request):
    """
    Function to run a classification process over the cloud based data
    # Parameters:
        - `TYPE` : `str` = type of classification to perform. Can be one of:
            + `RELEVANCE`   (tells if an article is relevant or not)
            + `SERVICE`     (tells which kind of service is about)
    # Returns:
    json-like object with the following fields:
        - `status` = One of>
            + `error`
            + `success`
    # Environment
    This function expects the following environment variables to be present:
        - `TABLE` : `str` = name of the table where to store and read news from
        - `PROJECT_ID` : `str` = project id to use when configuring firestore
    """
    # Get logging objects
    logging_client = logging.Client()
    logger = logging_client.logger("microscope-classify")

    # Parse config 
    config = ClassifierConfig(request)
    if config.error:
        logger.log_text(config.error, severity="ERROR")
        print(config.error)
        return flask.jsonify({"status":"error", "msg" : config.error})

class ClassifierConfig:
    """
        Simple object to parse configurations for a classifier
    """
    def __init__(self, request : flask.Request):

        # Parse table name fron environment variables
        table_name = os.environ.get("TABLE")
        if not table_name:
            self._error = "No TABLE provided in environment variables"
            return

        # Parse project id for firestore
        project_id = os.environ.get("PROJECT_ID")
        if not project_id:
            self._error = "No PROJECT_ID  provided in environment variables"
            return

        # set up driver data
        self._scrape_data_table_name = table_name
        client = ClassifierConfig.get_client()
        self._manager = Manager(BigQueryManager(table_name, client, project_id))

        self._error = None

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