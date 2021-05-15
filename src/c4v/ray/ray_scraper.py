"""
    functions to scrape urls in a distributed manner
"""

# External imports
from numpy.core.numeric import extend_all
import ray

# Local imports
import c4v.scraper.utils as scrp_utils
from c4v.scraper.scraper  import bulk_scrape
from c4v.scraper.scraped_data_classes.scraped_data import ScrapedData

# Python imports
from typing import List

@ray.remote
def _ray_scrape(urls : List[str]):
    return bulk_scrape(urls)

def ray_scrape(urls : List[str]) -> List[ScrapedData]:
    """
        Scrape a list of urls in ray-based distributed manner
    """

    # sort urls by domain
    urls.sort(key=scrp_utils.get_domain_from_url)
    
    # group urls by domain
    futures = [_ray_scrape.remote(url_list) for url_list in scrp_utils.group_by(urls, scrp_utils.get_domain_from_url)]
    
    # collect output
    output = []
    for l in ray.get(futures):
        output.extend(l)

    return output

