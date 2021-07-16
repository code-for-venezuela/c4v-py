"""
    Provide functionality to crawl urls for known sites in a ray-based manner
"""

# External imports
import ray
from ray.worker import remote
from c4v.scraper.crawler import crawlers

# Local imports
from c4v.scraper.settings import INSTALLED_CRAWLERS
from c4v.scraper.crawler.crawlers.base_crawler import BaseCrawler

# Python imports
from typing import Callable, Any, List

@ray.remote
def _ray_crawl_and_process( crawler : BaseCrawler, post_process_function : Callable[[List[str]], Any] ):
    """
        Crawl urls using crawler 'crawler', and then process them using the 'post_process_function'
        function
        Parameters:
            + crawler : BaseCrawler = Crawler instance to use to crawl for urls
            + post_process_function : [str] -> Any = Function to call over each batch of urls once it's scraped
    """
    crawler.crawl_and_process_urls(post_process_function)

def _ray_crawl(crawler : BaseCrawler) -> List[str]:
    """
        Retrieve urls crawled by given crawler instance
        Parameters:
            + crawler : BaseCrawler = Crawler instance to use for scraping
    """
    items = []

    def store_urls(urls : List[str]):
        items.extend(urls)

    _ray_crawl_and_process(crawler, store_urls)

def ray_crawl() -> List[str]:
    """
        Crawl a list of urls using all the available crawlers
    """

    futures = [_ray_crawl.remote(crawler_class()) for crawler_class in INSTALLED_CRAWLERS ]
    results = []

    for l in ray.get(futures):
        results.extend(l)

    return results
