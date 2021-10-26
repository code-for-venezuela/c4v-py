"""
    Base class for a scrapper.
    In order to create and wire a new scrapper:
        1) Create a new scraper in the "scrapers" directory
        2) Make your scraper a subclass of BaseScraper
        3) Implement missing methods (parse & scrape)
        4) add an entry in settings.py to the INSTALLED_SCRAPERS list. 
            Import it if necessary
"""

# Local imports
from c4v.scraper.scraped_data_classes.base_scraped_data import BaseDataFormat

# Python imports
from typing import List


class BaseScraper:
    """
        Base class for scrapers implementations
    """

    # domain to be scraped by this scraper
    intended_domain: str = None

    def parse(self, response) -> BaseDataFormat:
        """
            return scraped data from a response object 
            Parameters:
                + response : any = some kind of structure holding an http response
                                   from which we can scrape data
            Return:
                A dict with scrapped fields from response
        """
        raise NotImplementedError("Scrapers should implement the parse method")

    def scrape(self, url: str) -> BaseDataFormat:
        """
            return scraped data from url.
            Parameters: 
                + url : str = url to be scraped by this class
            Return:
                A dict with scrapped data from the given url
                if such url is a valid one
        """
        raise NotImplementedError("Scrapers should implement the scrape method")

    def bulk_scrape(self, urls: List[str]) -> List[BaseDataFormat]:
        """
            Return scraped data for a list of urls. Override it 
            if your scraper implementation could handle an optimized
            bulk scraping.

            Parametes:
                + urls : [str] = urls to be scraped
            Return:
                List of scraped items. Notice that the order it's not guaranteed to be
                the same as in the input list.
        """
        self.schedule_scraping(urls)
        self.start_bulk_scrape()
        return self.get_scraped_items()

    def schedule_scraping(self, urls: List[str]):
        """
            Schedule a list of urls to be scraped later.
            This is necessary so you can scrape data from multiple
            sources at the same time with multiple scrapers.

            Parameters:
                + urls : [str] = urls to be scheduled to scrape
        """
        self._to_scrape = urls

    def start_bulk_scrape(self):
        """
            Start a bulk scraping process, storing results internally
        """

        # Consistency check: cannot bulk scrape without something to scrape
        assert hasattr(
            self, "_to_scrape"
        ), "There's no scheduled items to scrape. Call schedule_scraping first to set items to be scraped"
        items = []

        for url in self._to_scrape:
            item = self.scrape(url)
            if item:
                items.append(item)

        del self._to_scrape
        self._scraped_items = items

    def get_scraped_items(self) -> List[BaseDataFormat]:
        """
            Return the scraped list of items after a bulk scrape process
        """
        assert hasattr(
            self, "_scraped_items"
        ), "There's no scraped items. Call start_bulk_scrape first to get items to retrieve"

        out = self._scraped_items
        del self._to_scrape

        return out
