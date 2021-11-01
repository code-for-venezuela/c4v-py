"""
    Main module interface
"""

# Local imports
from c4v.scraper.scraped_data_classes.base_scraped_data import BaseDataFormat
from c4v.scraper.scrapers.base_scraper import BaseScraper
from c4v.scraper.scraped_data_classes.scraped_data import ScrapedData
from .settings import URL_TO_SCRAPER
from c4v.scraper.utils import get_domain_from_url, valid_url

# Python imports
from typing import List, Type, Dict


def scrape(url: str) -> ScrapedData:
    """
        Scrape data for the given url if such url is scrappable,
        Raise ValueError if not. 

        Parameters:
            + url - str : Url to scrape
        Return:
            A ScrapedData object describing the data that could be 
            extracted for this url. Obtained data depends on the url itself, 
            so available data may change depending on the scrapped url, some fields 
            may be null.
    """
    scraper = _get_scraper_from_url(url)()
    return scraper.scrape(url)


def bulk_scrape(urls: List[str]) -> List[ScrapedData]:
    """
        Performs a bulk scraping over a list of urls.
        Order in the item list it's not guaranteed to be
        the same as in the input list

        Parameters:
            + urls : [str] = Urls to be scraped
        Return:
            A list of items scraped for each url in the original list
    """

    scrapers: Dict[Type[BaseScraper], List[str]] = {}

    # Classify urls to its according scraper
    for url in urls:
        scraper = _get_scraper_from_url(url)

        url_list = scrapers.get(scraper)
        if not (url_list):
            url_list = scrapers[scraper] = []

        url_list.append(url)

    # Schedule scraping
    scrapers_instances: List[BaseScraper] = []
    for (scraper, url_list) in scrapers.items():
        s = scraper()  # Create a new scraper instance
        s.schedule_scraping(url_list)
        scrapers_instances.append(s)

    # TODO: write a cleaner and transparent interface to show that scrapy scrapers only blocks once

    # start scraping for every scraper
    for s in scrapers_instances:
        s.start_bulk_scrape()

    # Retrieve scraped items
    items: List[BaseDataFormat] = []
    for s in scrapers_instances:
        items.extend(s.get_scraped_items())

    for i in range(len(items)):
        items[i] = items[i].to_scraped_data()
    return items


def _get_scraper_from_url(url: str) -> Type[BaseScraper]:
    """
        Validates if this url is scrapable and returns its 
        corresponding spider when it is.
        Raise ValueError if url is not valid or if it's not 
        scrapable for our supported scrapers
    """

    if not valid_url(url):
        raise ValueError(f"This is not a valid url: {url}")

    domain = get_domain_from_url(url)
    scraper = URL_TO_SCRAPER.get(domain)
    if not (scraper):
        raise ValueError(f"Unable to scrap this url: {url}")

    return scraper
