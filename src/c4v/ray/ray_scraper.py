"""
    functions to scrape urls in a distributed manner
"""

# External imports
import ray

# Local imports
import c4v.scraper.utils as scrp_utils
from c4v.scraper.scraper  import bulk_scrape
from c4v.scraper.scraped_data_classes.scraped_data import ScrapedData

# Python imports
from typing import List

ray.init()

@ray.remote
def _ray_scrape(urls : List[str]) -> List[ScrapedData]:
    """
        Create a new task with a scraping function
    """
    return bulk_scrape(urls)

def ray_bulk_scrape(urls : List[str], workers_amount : int = 3) -> List[ScrapedData]:
    """
        Scrape a list of urls in ray-based distributed manner
        Parameters:
            urls : [str] = list of urls to scrape
            workers_amount : int = ammount of workers to use
        Return:
            Data obtained after scraping
    """
    # sanity check
    assert workers_amount > 0, "Should be at the least 1 worker"
    
    # group urls by domain
    futures = [_ray_scrape.remote(url_list) for url_list in scrp_utils.generate_chunks(urls, workers_amount)]
    
    # collect output
    output = []
    for l in ray.get(futures):
        output.extend(l)

    return output
