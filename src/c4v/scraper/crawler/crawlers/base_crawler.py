"""
    Base class for url crawlers. Based on sitemap crawling.
"""
# Python imports
from typing import Any, List, Callable
import sys
import re

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
    name: str = None  # Crawler name, required to identify this crawler
    ALL_URLS = [".*"]
    NO_URLS = ["a^"]
    IRRELEVANT_URLS = []

    def __init__(
        self, white_list: List[str] = None, black_list: List[str] = None
    ) -> None:
        self._black_list = black_list or self.NO_URLS
        self._white_list = white_list or self.ALL_URLS

    @staticmethod
    def _to_regex(patterns: List[str]) -> str:
        """
            Convert given list of regex to a single regex 
        """
        return "(" + ")|(".join(patterns) + ")"

    @property
    def white_list_regex(self) -> str:
        """
            Regex matching every white listed url regex pattern
        """
        return self._to_regex(self._white_list)

    @property
    def black_list_regex(self) -> str:
        """
            Regex matching every black listed url regex pattern
        """
        return self._to_regex(self._black_list)

    def crawl_urls(self, up_to: int = None) -> List[str]:
        """
            Return a list of urls scraped from the site intended for this scraper.
            Parameters:
                up_to : int = Maximum amount of elements to store. Retrieve all if no number is provided, should be positive 
            Return:
                List of urls from sitemap
        """

        # Set up max size
        up_to = up_to or sys.maxsize

        # Check for consistency
        if up_to <= 0:
            raise ValueError("Max size should be a possitive number")

        # Set up storing function
        items = []

        def store_items(new_items: List[str]):
            rem = up_to - len(items)
            items.extend(new_items[:rem])

        # Set up stop function
        def should_stop_when() -> bool:
            return len(items) >= up_to

        # crawl for urls
        self.crawl_and_process_urls(store_items, should_stop_when)

        return items

    def crawl_and_process_urls(
        self,
        post_process_data: Callable[[List[str]], Any] = None,
        should_stop: Callable[[], bool] = None,
    ):
        """
            crawl urls, processing them with the provided function
            Parameters:
                + post_process_data : ([str]) -> [str] = function to call over the resulting set of urls. May be called in batches
                + should_stop : () -> bool = function to every iteration to check if the crawling process should stop
            Return:
                List of urls from sitemap
        """

        # Url accumulator
        urls: List[str] = []

        # sitemaps to parse
        sitemaps = self.get_sitemaps_from_index()

        # Set up stop function
        should_stop = should_stop or (lambda _: False)

        for sitemap in sitemaps:

            # Get a sitemap from its url
            sitemap_content = self.get_sitemap_from_url(sitemap)

            # parse urls for the current sitemap
            new_urls = self.parse_urls_from_sitemap(sitemap_content)
            urls.extend(new_urls)

            # process new urls if post process function exists
            if post_process_data:
                post_process_data(new_urls)

            if should_stop():
                break

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
        # We request the white_list regex once and use it often because otherwise, such string will be
        # computed once per url, which is quite inneficient
        white_list = self.white_list_regex
        urls = [
            u
            for u in urls
            if self.should_scrape(u) and self.is_white_listed(u, white_list)
        ]

        return urls

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

    def is_white_listed(self, url: str, white_list_regex: str = None) -> bool:
        """
            Checks if the given url as string matches list of white listed patterns
        """
        return not not re.match(white_list_regex or self.white_list_regex, url)

    @classmethod
    def from_irrelevant(cls):
        """
            Return a crawler created for filtering only irrelevant urls
        """
        return cls(cls.IRRELEVANT_URLS)
