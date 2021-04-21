"""
    Base class for scrapy-based scrapers.

    In order to create a a new scrapy scraper:
        1) Create a new scraper un "scrapers" folder, and make it subclass
            of this BaseScrapyScraper
        2) override "spider" attribute of your new class with a valid
            scrapy spider
        3) wired it in settings as you would do with a regular scraper 
"""

# External imports
from scrapy import Spider

# Internal imports
from scraper.scrapers.base_scraper import BaseScraper
from scraper.spider_manager import SpiderManager

# Python imports
from typing import Type, List


class BaseScrapyScraper(BaseScraper):
    """
        In order to create a new Scrappy Scrapper, just 
        inherit this class and assign a new value to the 
        "spider" field, a valid scrapy Spider sub class.
    """

    spider: Type[Spider] = None

    def __init__(self):

        if self.spider is None:
            raise TypeError(
                "Spider not defined,"
                + "perhaps you forgot to override spider"
                + "attribute in BaseScrapyScraper subclass?"
            )

        self._spider_manager = SpiderManager(self.spider)

    def parse(self, response) -> dict:
        # Delegate parse function to scrapy implementation
        return self._spider_manager.parse(response)

    def scrape(self, url: str) -> dict:
        # Delegate scrape function to scrapy implementation
        return self._spider_manager.scrape(url)

    def bulk_scrape(self, urls: List[str]) -> List[dict]:
        # Delegate bulk_scrape function to scrapy implementation
        return self._spider_manager.bulk_scrape(urls)
