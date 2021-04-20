"""
    This file exports every spider or scrapper implementation for each site.
"""
# external imports
import scrapy
from scrapy.crawler import CrawlerProcess
import scrapy.signals

# Project imports
import scraper.utils    as utils
import scraper.settings as settings



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

    def scrape(self, url : str) -> dict:
        """
            Return scraped data from a single Url
            Parameters:
                + url : str = url whose data is to be scraped. Should be compatible with the given spider
            Return:
                dict like object with scraped data
        """
        scraped = self.scrape_all([url])

        return scraped[0] if scraped else None


    def scrape_all(self, urls : List[str]) -> List[dict]: 
        """
            return scraped data from a list of valid URLs
            Parameters:
                + urls : [str] = urls whose data is to be scraped. 
                                Should be compatible with the provided spider
            Return:
                list of dict like object with scraped data
        """

        # if nothing to do, just return an empty list
        if not urls: return []

        # Items accumulator
        items = []


        # callback function to collect items on the fly
        def items_scrapped(item, response, spider):
            items.append({"url": response._url, "data": item})

        # set up urls to scrape
        self.spider.start_urls = urls

        # create crawler for this spider, connect signal so we can collect items
        crawler = self.process.create_crawler(self.spider)
        crawler.signals.connect(items_scrapped, signal= scrapy.signals.signals.item_scraped)

        # start scrapping
        self.process.crawl(crawler)
        self.process.start()

        # return post processed scrapped objects
        return items



class ElPitazoSpider(scrapy.Spider):
    """
        Spider to scrape ElPitazo data 
    """

    name = "el_pitazo"

    start_urls = []

    def parse(self, response):
        """
            Returns a dict like structure with the following 
            fields:
                + title
                + date
                + categories
                + body
                + author 
                + tags
        """

        # These are simple properties, just get its text with a valid
        # selector
        title = utils.get_element_text(".tdb-title-text", response) or ""
        date = utils.get_element_text(".entry-date", response) or ""
        author = utils.get_element_text(".tdb-author-name", response) or ""

        body = self._get_body(response)

        tags = self._get_tags(response)

        # categories
        categories = response.css(".tdb-entry-category").getall()
        categories = list(map(utils.clean, categories))

        return {
            "title": title,
            "date": date,
            "categories": categories,
            "body": body,
            "author": author,
            "tags": tags,
        }

    def _get_body(self, response) -> str:
        """
            Get article body as a single string
        """
        body = response.css("#bsf_rt_marker > p").getall()
        body = map(utils.clean, body)
        body = "".join(body)

        return body

    def _get_tags(self, response) -> List[str]:
        """
            Try to get tags from document if available
        """
        tags = response.css(".tdb-tags > li > a").getall()
        tags = list(map(utils.clean, tags))
        return tags
