"""
    Main module interface
"""

# Local imports
from scraper.scrapers.base_scraper import BaseScraper
from .settings import URL_TO_SCRAPER
from scraper.utils import get_domain_from_url, valid_url

# Python imports
from typing import List, Type


def scrape(url: str) -> dict:
    """
        Scrape data for the given url if such url is scrappable,
        Raise ValueError if not. 

        Params:
            + url - str : Url to scrape
        Return:
            A dict object, each describing the data that could be 
            extracted for this url. Obtained data depends on the url itself, 
            so available data may change depending on the scrapped url.
            Dict format:
             {
                 "url" : (str) url where the data came from,
                 "data": (dict) Data scraped for this url
             }
    """
    scraper = _get_scraper_from_url(url)()
    return scraper.scrape(url)


def bulk_scrape(urls: List[str]) -> List[dict]:
    """
        Performs a bulk scraping over a list of urls.
        Order in the item list it's not guaranteed to be
        the same as in the input list

        Parameters:
            + urls : [str] = Urls to be scraped
        Return:
            A list of items scraped for each url in the original list
    """

    items = []
    scrapers = {}
    for url in urls:
        # Classify urls to its according scraper
        scraper = _get_scraper_from_url(url)

        if not (url_list := scrapers.get(scraper)):
            url_list = scrapers[scraper] = []

        url_list.append(url)

    # Bulk scrape urls
    for (scraper, url_list) in scrapers.items():
        s = scraper()  # Create a new scraper instance
        items.extend(s.bulk_scrape(url_list))

    return items


def _get_scraper_from_url(url: str) -> Type[BaseScraper]:
    """
        Validates if this url is scrapable and returns its 
        corresponding spider when it is
    """

    if not valid_url(url):
        raise ValueError(f"This is not a valid url: {url}")

    domain = get_domain_from_url(url)

    if not (scraper := URL_TO_SCRAPER.get(domain)):
        raise ValueError(f"Unable to scrap this url: {url}")

    return scraper
