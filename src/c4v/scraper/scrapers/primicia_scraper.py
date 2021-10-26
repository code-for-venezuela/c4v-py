"""
    Scraper to get data from Primicia
"""
# Internal imports
from c4v.scraper.scrapers.base_scrapy_scraper import BaseScrapyScraper
from c4v.scraper.spiders.primicia import PrimiciaSpider


class PrimiciaScraper(BaseScrapyScraper):
    """
        Scrapes data from ElPitazo, relies in 
        scrapy for this.
    """

    intended_domain = "primicia.com.ve"
    spider = PrimiciaSpider
