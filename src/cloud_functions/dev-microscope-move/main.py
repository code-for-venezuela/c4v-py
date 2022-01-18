# Python imports
from typing import Dict
import os

# Local imports
from  c4v.scraper.persistency_manager.big_query_persistency_manager import BigQueryManager

# Third party imports
import flask
from google.cloud import  bigquery, logging

def move(request : flask.Request):
    """
        This function will move data from firestore to big query once it's complete
        # Return
            If was able to perform move
        # Environment
        This function expects the following environment variables to be present:
        - `TABLE` : `str` = name of the table where to store and read news from
        - `PROJECT_ID` : `str` = project id to use when configuring firestore
    """
    # Get logging objects
    logging_client = logging.Client()
    logger = logging_client.logger("microscope-move")

    config = MoveConfig(request)
    if config.error:
        logger.log_text(config.error, severity="ERROR" )
        return {"status" : "error", "msg" : config.error}
    
    config.manager.move()

    return {"status" : "success"}

class MoveConfig:
    """
        This class holds the set up for the move function
    """

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


        # set up driver data
        self._scrape_data_table_name = table_name
        client = MoveConfig.get_client()
        self._manager = BigQueryManager(table_name, client, project_id)

        self._error = None

    @property
    def manager(self) -> BigQueryManager:
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
