import logging
from google.cloud import bigquery
from google.oauth2 import service_account
import os

logger = logging.getLogger(__name__)

SERVICE_ACCOUNT_KEY_PATH_ENV_VAR = "ANGOSTURA_SERVICE_ACCOUNT"


class AngosturaLoader:
    def __init__(self, service_account_key_path=None):
        """
        Creates connection to perform queries to Code For Venezuela's Angostura ETL.
        BigQuery is used as warehouse.
        
        Parameters:
        ------

            service_account_key_path: String, Default = None.
                Path to service account json key. 
                This json key is created by the developers of Angostura, to have 
                access to this key please contact the C4V team.
        
        Attributes:
        ----------

            client: google.cloud.bigquery.client.Client object.
                Connection object to Angostura.

            key: String.
                Path to service account json key. 

        Examples:
        ---------
        
            # Note - It is necessary to have the key angostura_connection.json in the 
            # root directory. If you don't have it please contact anyone from c4v's Sambil team.
            test = AngosturaLoader()
            
            df = test.create_query(
                    "SELECT * FROM `event-pipeline.angostura.sinluz_rawtweets` LIMIT 1"
            )
            
            print(df.head())
        """
        self.client = None

        if service_account_key_path:
            logger.info("Using user input key path.")
            self.key_path = service_account_key_path
        elif SERVICE_ACCOUNT_KEY_PATH_ENV_VAR in os.environ:
            logger.info("Using environment variable.")
            self.key_path = os.environ[SERVICE_ACCOUNT_KEY_PATH_ENV_VAR]
        else:
            logger.info("Looking for key in project's root directory.")
            self.key_path = "angostura_connection.json"

    def create_connection(self):
        """
        Creates connection with angostura.
        
        Parameters:
        ----------

            Self: Object.
                Object instance.
        
        Returns:
        --------

            self.client: google.cloud.bigquery.client.Client object
                Connection object to Angostura.
        """

        credentials = service_account.Credentials.from_service_account_file(
            self.key_path, scopes=["https://www.googleapis.com/auth/cloud-platform"],
        )

        self.client = bigquery.Client(
            credentials=credentials, project=credentials.project_id,
        )

    def create_query(self, query):
        """
        Query Angostura's warehouse.
        
        Parameters:
        ----------

            Query: String.
                SQL query with BigQuery syntax.

        Returns:
        --------
            results_df: Pandas.DataFrame n,n.
                Pandas Dataframe with query results.
        """

        if self.client is None:
            self.create_connection()

        logger.info("Querying Database...")
        QUERY = query

        results_df = self.client.query(QUERY).to_dataframe()

        return results_df
