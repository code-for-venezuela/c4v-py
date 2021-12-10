import flask
from c4v.microscope import Manager 
from google.cloud import bigquery
from c4v.scraper.persistency_manager.big_query_persistency_manager import BigQueryManager, ScrapedData
import os

def get_client() -> bigquery.Client:
    client = bigquery.Client()
    return client 

def crawl(request : flask.Request):
    """
        function to trigger a crawling process in google cloud
    """
    data = ScrapedData(url="www.michinews.com", title="sueñan los gatos con ratones eléctricos", content="No, no sueñan con ratones eléctricos", author="michiberto rodríguez", categories=["gatos", "sueños", "ratones eléctricos"])

    client = get_client()
    project_id = "event-pipeline"
    dataset_name = "sambil_collab_dev"
    scrape_table = f"{project_id}.{dataset_name}.scraped_data"
    x = "event-pipeline.sambil_collab_dev.scraped_data"
    table_name = os.environ.get("TABLE", scrape_table)
    bq_manager = BigQueryManager(table_name, client)
    bq_manager.save([data])
    ret = bq_manager.get_all()


    return str([x for x in ret])