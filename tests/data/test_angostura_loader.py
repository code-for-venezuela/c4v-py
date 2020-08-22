from c4v.data.angostura_loader import AngosturaLoader
from google.cloud import bigquery
from google.oauth2 import service_account

def test_angostura():

    test = AngosturaLoader(service_account_key_path = "./event-pipeline-beac3494771d.json")
    df = test.create_query("SELECT * FROM `event-pipeline.angostura.sinluz_rawtweets` LIMIT 1")
    df.info()