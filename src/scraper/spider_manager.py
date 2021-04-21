# external imports
import scrapy
from scrapy.crawler import CrawlerProcess
import scrapy.signals

# Project imports
import scraper.scrapy_settings as settings

# Python imports
from typing import List


class SpiderManager:
    """
        Utility class to perform common operations in 
        Spider classes
    """

    process = CrawlerProcess(settings.CRAWLER_SETTINGS)

    def __init__(self, spider) -> None:
        self.spider = spider
        pass

    def parse(self, response) -> dict:
        """
            return scraped data from a valid response
            Parameters: 
                + response : scrapy.http.Response = response object holding the actual response
            Return:
                dict like object with scraped data
        """
        spider = self.spider()
        return spider.parse(response)

    def scrape(self, url: str) -> dict:
        """
            Return scraped data from a single Url
            Parameters:
                + url : str = url whose data is to be scraped. Should be compatible with the given spider
            Return:
                dict like object with scraped data
        """
        scraped = self.bulk_scrape([url])

        return scraped[0] if scraped else None

    def bulk_scrape(self, urls: List[str]) -> List[dict]:
        """
            return scraped data from a list of valid URLs
            Parameters:
                + urls : [str] = urls whose data is to be scraped. 
                                Should be compatible with the provided spider
            Return:
                list of dict like object with scraped data
        """

        # if nothing to do, just return an empty list
        if not urls:
            return []

        # Items accumulator
        items = []

        # callback function to collect items on the fly
        def items_scrapped(item, response, spider):
            items.append({"url": response._url, "data": item})

        # set up urls to scrape
        self.spider.start_urls = urls

        # create crawler for this spider, connect signal so we can collect items
        crawler = self.process.create_crawler(self.spider)
        crawler.signals.connect(items_scrapped, signal=scrapy.signals.item_scraped)

        # start scrapping
        self.process.crawl(crawler)
        self.process.start()

        # return post processed scrapped objects
        return items
