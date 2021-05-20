"""
    Base class for url crawlers. Based on sitemap crawling.
"""
# Python imports
from typing import Any, List, Callable

class BaseCrawler:
    """
        Inherit this class to create a new crawler.
    """

    start_sitemap_url : str = None # Override this field to define sitemap to crawl

    def crawl_urls(self, post_process_data : Callable[[List[str]], Any] = None) -> List[str]:
        """
            Return a list of urls scraped from the site intended for this scraper.
            Parameters:
                + post_process_data : ([str]) -> [str] = function to call over the resulting set of urls. May be called in batches
            Return:
                List of urls from sitemap
        """

        # Url accumulator
        urls : List[str] = []

        # sitemaps to parse
        sitemaps = self.get_sitemaps()

        for sitemap in sitemaps:
            # parse urls for the current sitemap
            urls.extend(new_urls := self.parse_urls_from_sitemap(sitemap))

            # process new urls if post process function exists
            if post_process_data:
                post_process_data(new_urls)


        return urls

    def get_sitemaps(self) -> List[str]:
        """
            Some sites may have its sitemap paginated in more sitemaps.
            I such case, override this class. By default, it assumes 
            the sitemap set is actually the starting one.
        """
        return [self.start_sitemap_url]

    def parse_urls_from_sitemap(self, sitemap : str) -> List[str]:
        """
            Given a sitemap url, parse site urls from it 
        """
        raise NotImplementedError("Implement parse_urls_from_sitemap from abstract class BaseCrawler")

