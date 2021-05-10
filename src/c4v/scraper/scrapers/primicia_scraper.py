"""
    Scraper to get data from Primicia
"""
# Internal imports
from c4v.scraper.scrapers.base_scrapy_scraper import BaseScrapyScraper
from c4v.scraper.spiders.primicia import PrimiciaSpider
from c4v.scraper.scraped_data_classes.elpitazo_scraped_data import ElPitazoData


class PrimiciaScraper(BaseScrapyScraper):
    """
        Scrapes data from ElPitazo, relies in 
        scrapy for this.
    """

    intended_domain = "primicia.com.ve"
    spider = PrimiciaSpider
