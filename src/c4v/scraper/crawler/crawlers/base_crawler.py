"""
    Base class for url crawlers. Based on sitemap crawling.
"""
# Python imports
from typing import Any, List, Callable

# Third party imports
import requests
import bs4


class BaseCrawler:
    """
        Inherit this class to create a new crawler.
        Probably, the only function you might want to implement is check_sitemap_url,
        that checks if an url in the sitemap index corresponds to an interesting subset 
        of pages

        You might also want to override should_scrape method if you want to add detailed 
        filtering to over urls to be retrieved 
    """

    start_sitemap_url: str = None  # Override this field to define sitemap to crawl
    name: str = None               # Crawler name, required to identify this crawler 

    def crawl_urls(self) -> List[str]:
        """
            Return a list of urls scraped from the site intended for this scraper.
            Return:
                List of urls from sitemap
        """
        items = []
        def store_items(new_items : List[str]):
            items.extend(new_items)

        self.crawl_and_process_urls(store_items)

        return items

    def crawl_and_process_urls(
        self, post_process_data: Callable[[List[str]], Any] = None
    ):
        """
            crawl urls, processing them with the provided function
            Parameters:
                + post_process_data : ([str]) -> [str] = function to call over the resulting set of urls. May be called in batches
            Return:
                List of urls from sitemap
        """

        # Url accumulator
        urls: List[str] = []

        # sitemaps to parse
        sitemaps = self.get_sitemaps_from_index()

        for sitemap in sitemaps:

            # Get a sitemap from its url
            sitemap_content = self.get_sitemap_from_url(sitemap)

            # parse urls for the current sitemap
            new_urls = self.parse_urls_from_sitemap(sitemap_content)
            urls.extend(new_urls)

            # process new urls if post process function exists
            if post_process_data:
                post_process_data(new_urls)

        return urls

    def get_sitemaps_from_index(self) -> List[str]:
        """
            Some sites may have its sitemap paginated in more sitemaps.
            I such case, override this class. By default, it assumes 
            the sitemap set is actually the starting one.
        """
        assert self.start_sitemap_url != None, "Start sitemap url not configured"

        resp = requests.get(self.start_sitemap_url)
        if resp.status_code != 200:
            resp.raise_for_status()

        return self.parse_sitemaps_urls_from_index(resp.text)

    def parse_sitemaps_urls_from_index(self, sitemap_index: str) -> List[str]:
        """
            Get sitemap list from the content of an xml file with the filemap data
            Parameters:
                + sitemap_index : str = sitemap index xml content as string
            Return:
                List of urls listed by this index
        """
        soup = bs4.BeautifulSoup(sitemap_index, "xml")
        urls = map(lambda l: l.get_text(), soup.find_all("loc"))
        urls = filter(self.should_crawl, urls)

        return list(urls)

    def get_sitemap_from_url(self, sitemap_url: str) -> str:
        """
            Get sitemap xml content from its corresponding url
            Parameters:
                + sitemap_url : str = sitemap url to retrieve
            Return:
                sitemap content if everything went ok
        """
        resp = requests.get(sitemap_url)
        if resp.status_code != 200:
            resp.raise_for_status()

        return resp.text

    def parse_urls_from_sitemap(self, sitemap: str) -> List[str]:
        """
            Given a sitemap body, parse site urls from it 
            Parameters:
                + sitemap : str = sitemap xml content as string
            Return:
                list of urls parsed for this sitemap
        """
        soup = bs4.BeautifulSoup(sitemap, "xml")
        urls = map(lambda l: l.get_text(), soup.select("url > loc"))
        urls = filter(self.should_scrape, urls)

        return list(urls)

    @staticmethod
    def should_crawl(url: str) -> bool:
        """
            Function to check if a given sitemap url
            to another sitemap is a desired one
            Parameters:
                + url : str = url to check
            return:
                if this is a valid url
        """
        raise NotImplementedError("Implement this abstract method")

    @staticmethod
    def should_scrape(url: str) -> bool:
        """
            Function to check if an url to a web page in the site is a valid one
            Parameters:
                + url : str = a web page url to check
            Return:
                boolean, telling if this is a valid url
        """
        return True
