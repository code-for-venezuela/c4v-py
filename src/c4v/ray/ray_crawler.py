"""
    Provide functionality to crawl urls for known sites in a ray-based manner
"""

# External imports
import ray

# Local imports
from c4v.scraper.settings import INSTALLED_CRAWLERS
from c4v.scraper.crawler.crawlers.base_crawler import BaseCrawler

# Python imports
from typing import Callable, Any, List

@ray.init()

@ray.remote
def _ray_crawl_and_process_urls( crawler : BaseCrawler, post_process_function : Callable[[List[str]], Any], should_stop : Callable[[], bool]):
    """
        Crawl urls using crawler 'crawler', and then process them using the 'post_process_function'
        function
        Parameters:
            crawler : BaseCrawler = Crawler instance to use to crawl for urls
            post_process_function : [str] -> Any = Function to call over each batch of urls once it's scraped
    """
    crawler.crawl_and_process_urls(post_process_function, should_stop=should_stop)

@ray.remote
def _ray_crawl(crawler : BaseCrawler) -> List[str]:
    """
        Retrieve urls crawled by given crawler instance
        Parameters:
            crawler : BaseCrawler = Crawler instance to use for scraping
        Return:
            List of crawled urls
    """
    items = []

    def store_urls(urls : List[str]):
        items.extend(urls)

    _ray_crawl_and_process_urls(crawler, store_urls)

def ray_crawl_and_process_urls( crawlers : List[str], 
                                post_process_function : Callable[[List[str]], Any], 
                                should_stop : Callable[[], bool]
                            ):
    """
        Crawl a process urls, post processing them with the given function and 
        stoping when specified by the given predicate function.
        Parameters:
            crawlers : [str] = List of crawler names refering to the crawlers to be ran. Unknown crawlers
                                will be ignored
            post_process_function : ([str]) -> () = Function to post process crawled results in batch, don't expect every batch to be the same size 
            should_stop : () -> bool = Function to check if the crawler should keep crawling every iteration
    """
    futures = [ _ray_crawl_and_process_urls.remote(crawler_class(), post_process_function, should_stop) for crawler_class in INSTALLED_CRAWLERS if crawler_class.name in crawlers]
    for _ in ray.get(futures):
        pass

def ray_crawl(crawlers : List[str] = None) -> List[str]:
    """
        Crawl a list of urls using all the available crawlers.
        Parameters:
            crawlers : [str] = list of crawlers to be ran, unknown crawlers will be ignored
    """

    futures = [ _ray_crawl.remote(crawler_class()) for crawler_class in INSTALLED_CRAWLERS if crawler_class.name in crawlers]
    results = []

    for l in ray.get(futures):
        results.extend(l)

    return results
