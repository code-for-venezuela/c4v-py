"""
    Scraper to get data from El Pitazo
"""
# Internal imports
from c4v.scraper.scrapers.base_scrapy_scraper import BaseScrapyScraper
from c4v.scraper.spiders.el_pitazo import ElPitazoSpider


class ElPitazoScraper(BaseScrapyScraper):
    """
        Scrapes data from ElPitazo, relies in 
        scrapy for this.
    """

    intended_domain = "elpitazo.net"
    spider = ElPitazoSpider
