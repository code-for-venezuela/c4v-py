#%%
# Python imports
import os
from typing import Dict

# Local imports 
from c4v.microscope import Manager
from c4v.config import settings
from c4v.scraper.persistency_manager.big_query_persistency_manager import BigQueryManager
from c4v.cloud.gcloud_storage_manager import ClassifierType

# Third party imports
import flask
from google.cloud import bigquery, logging

def classify(request : flask.Request):
    """
    Function to run a classification process over the cloud based data
    # Parameters:
        - `type` : `str` = type of classification to perform. Can be one of:
            + `relevance`   (tells if an article is relevant or not)
            + `service`     (tells which kind of service is about)
    # Returns:
    json-like object with the following fields:
        - `status` = One of>
            + `error`
            + `success`
    # Environment
    This function expects the following environment variables to be present:
        - `TABLE` : `str` = name of the table where to store and read news from
        - `PROJECT_ID` : `str` = project id to use when configuring firestore
        - `C4V_STORAGE_BUCKET` : `str` = name of the bucket where to store and retrieve classifiers

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

    # Download classifier type 
    logger.log_text(f"Downloading classifier of type '{config.type}'")
    #Cloud function vm is a read only s/m. The only writable place is the tmp folder
    path = "/tmp/model"
    config.manager.download_model_to_directory(path, config.type)

    # Now the model will be stored in ./<type>, where <type> is the one provided as a function argument

    # DEBUG vamos a probar esto para ver si llegamos hasta aquí
    # TODO remember to put a return in here

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

        # Get type of classifier from request

        # Parse options from request
        request_json : Dict = request.get_json()
        classifier_type = request_json.get("type")

        # Parse classifier type
        if classifier_type not in ClassifierType.choices():
            self._error = f"Invalid `type` of classifier '{classifier_type}', choices are: {ClassifierType.choices()}"
            return
        
        self._type = classifier_type

        # set up driver data
        self._scrape_data_table_name = table_name
        client = ClassifierConfig.get_client()
        self._manager = Manager(BigQueryManager(table_name, client, project_id))

        self._error = None

    @staticmethod
    def _set_up_settings():
        settings.c4v_folder = "/tmp/c4v"
        settings.experiments_dir = "/tmp/c4v/experiments"


    @property
    def type(self) -> str:
        return self._type

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