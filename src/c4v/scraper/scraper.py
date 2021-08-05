"""
    Main module interface
"""

# Local imports
from c4v.scraper.scraped_data_classes.base_scraped_data import BaseDataFormat
from c4v.scraper.scrapers.base_scraper import BaseScraper
from c4v.scraper.scraped_data_classes.scraped_data import ScrapedData
from .settings import URL_TO_SCRAPER, INSTALLED_CRAWLERS
from c4v.scraper.utils import get_domain_from_url, valid_url
from c4v.scraper.persistency_manager.base_persistency_manager import (
    BasePersistencyManager,
)

# Python imports
from typing import List, Type, Dict
import sys


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


class Scraper:
    """
        This object automates handling scraping and crawling
    """

    def __init__(self, persistency_manager: BasePersistencyManager):
        self._persistency_manager = persistency_manager

    def get_bulk_data_for(self, urls: List[str], should_scrape: bool = True):
        """
            Retrieve scraped data for given url set if scrapable
            Parameters:
                urls : [str] = urls whose data is to be retrieved. If not available yet, then scrape it if requested so
                should_scrape : bool = if should scrape non-existent urls
        """
        # just a shortcut
        db = self._persistency_manager

        # Separate scraped urls from non scraped
        not_scraped = db.filter_scraped_urls(urls)

        # Scrape missing instances if necessary
        if should_scrape and not_scraped:
            items = bulk_scrape(not_scraped)
            db.save(items)

        # Convert to set to speed up lookup
        urls = set(urls)
        return [sd for sd in db.get_all() if sd.url in urls]

    def get_data_for(self, url: str, should_scrape: bool = True) -> ScrapedData:
        """
            Get data for this url if stored and scrapable. May return none if could not
            find data for this url
            Parameters:
                url : str = url to be scraped
                should_scrape : bool = if should scrape this url if not available in db
        """
        data = self.get_bulk_data_for([url], should_scrape)
        return data[0] if data else None

    def scrape_pending(self, limit: int = -1):
        """
            Update DB by scraping rows with no scraped data, just the url
            Parameters:
                limit : int = how much measurements to scrape, set a negative number for no limit
        """

        db = self._persistency_manager

        scrape_urls = [d.url for d in db.get_all(limit=limit, scraped=False)]

        scraped = bulk_scrape(scrape_urls)

        db.save(scraped)

    def crawl_new_for(self, crawler_names: List[str] = None):
        """
            Crawl for new urls using the given crawlers only
            Parameters:
                crawler_names : [str] = names of crawlers to be ran when this function is called. If no list is passed, then 
                                        all crawlers will be used
        """
        db = self._persistency_manager

        # Function to process urls as they come
        def save_urls(urls: List[str]):
            urls = db.filter_scraped_urls(urls)
            print(urls)
            datas = [ScrapedData(url=url) for url in urls]
            db.save(datas)

        # Names for installed crawlers
        crawlers = [c.name for c in INSTALLED_CRAWLERS]

        # if no list provided, default to every crawler
        if crawler_names == None:
            crawler_names = crawlers

        not_registered = [name for name in crawler_names if name not in crawlers]

        # Report warning if there's some non registered crawlers
        if not_registered:
            print(
                "WARNING: some names in given name list don't correspond to any registered crawler.",
                file=sys.stderr,
            )
            print(
                "Unregistered crawler names: \n"
                + "\n".join([f"\t* {name}" for name in not_registered])
            )

        # Instantiate crawlers to use
        crawlers_to_run = [
            crawler() for crawler in INSTALLED_CRAWLERS if crawler.name in crawler_names
        ]

        # crawl for every crawler
        for crawler in crawlers_to_run:
            crawler.crawl_and_process_urls(save_urls)
