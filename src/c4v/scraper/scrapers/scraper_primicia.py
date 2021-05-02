"""
    Scraper to get data from Primicia
"""
# Internal imports
from c4v.scraper.scrapers.base_scrapy_scraper import BaseScrapyScraper
from c4v.scraper.spiders.spider_primicia import SpiderPrimicia


class ScraperPrimicia(BaseScrapyScraper):
    """
        Scrapes data from Primicia, relies in
        scrapy for this.
    """

    spider = SpiderPrimicia
