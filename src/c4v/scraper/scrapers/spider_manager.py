# external imports
import scrapy
from scrapy.crawler import CrawlerProcess
import scrapy.signals

# Project imports
from . import scrapy_settings as settings
from c4v.scraper.scraped_data_classes.scraped_data import ScrapedData

# Python imports
from typing import List


class SpiderManager:
    """
        Utility class to perform common operations in 
        Spider classes
    """

    _process = None

    @classmethod
    def process(cls) -> CrawlerProcess:
        """
            retrieves process object
        """
        if not cls._process:
            cls._process = CrawlerProcess(settings.CRAWLER_SETTINGS)
        return cls._process

    def __init__(self, spider) -> None:

        self.spider = spider
        self._scraped_items = []

        def add_item(item):
            self._scraped_items.append(item)

        self._add_items = add_item

    def parse(self, response) -> ScrapedData:
        """
            return scraped data from a valid response
            Parameters: 
                + response : scrapy.http.Response = response object holding the actual response
            Return:
                dict like object with scraped data
        """
        spider = self.spider()
        return spider.parse(response)

    def scrape(self, url: str) -> ScrapedData:
        """
            Return scraped data from a single Url
            Parameters:
                + url : str = url whose data is to be scraped. Should be compatible with the given spider
            Return:
                dict like object with scraped data
        """
        self.schedule_scraping([url])
        self.start_bulk_scrape()
        scraped = self.get_scraped_items()
        return scraped[0] if scraped else None

    def schedule_scraping(self, urls: List[str]):
        """
            Schedule urls to be scraped
            Parameters:
                + urls : [str] = list of urls to scrape
        """
        # if nothing to do, just return an empty list
        if not urls:
            return

        # set up urls to scrape
        self.spider.start_urls = urls

        # create crawler for this spider, connect signal so we can collect items
        crawler = SpiderManager.process().create_crawler(self.spider)
        crawler.signals.connect(self._add_items, signal=scrapy.signals.item_scraped)

        # start scrapping
        SpiderManager.process().crawl(crawler)

    def start_bulk_scrape(self):
        """
            Scrape stored urls. Note that if multiple spider managers
            called schedule_scraping before, all of them will be scraped,
            not only this one
        """
        if SpiderManager.process().crawlers:
            SpiderManager.process().start()

    def get_scraped_items(self) -> List[ScrapedData]:
        """
            Get items scraped by a scraping process, flushing internal list in the 
            process
            Return:
                List of scraped elements 
        """
        scraped = self._scraped_items
        self._scraped_items = []
        return scraped
