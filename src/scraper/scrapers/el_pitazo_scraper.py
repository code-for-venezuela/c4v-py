"""
    Scraper to get data from El Pitazo
"""
# Internal imports
from scraper.scrapers.base_scrapy_scraper import BaseScrapyScraper
from scraper.spiders.el_pitazo import ElPitazoSpider


class ElPitazoScraper(BaseScrapyScraper):
    """
        Scrapes data from ElPitazo, relies in 
        scrapy for this.
    """

    spider = ElPitazoSpider
