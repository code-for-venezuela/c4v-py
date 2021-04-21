"""
    Base class for a scrapper.
    In order to create and wire a new scrapper:
        1) Create a new scraper in the "scrapers" directory
        2) Make your scraper a subclass of BaseScraper
        3) Implement missing methods (parse & scrape)
        4) add an entry in settings.py to the URL_TO_SCRAPER map, maping from 
           a domain name to your new scraper. Import it if necessary
"""

# Python imports
from typing import List


class BaseScraper:
    """
        Base class for scrapers implementations
    """

    def parse(self, response) -> dict:
        """
            return scraped data from a response object 
            Parameters:
                + response : any = some kind of structure holding an http response
                                   from which we can scrape data
            Return:
                A dict with scrapped fields from response
        """
        pass

    def scrape(self, url: str) -> dict:
        """
            return scraped data from url.
            Parameters: 
                + url : str = url to be scraped by this class
            Return:
                A dict with scrapped data from the given url
                if such url is a valid one
        """
        pass

    def bulk_scrape(self, urls: List[str]) -> List[dict]:
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

        items = []
        for url in urls:
            if (item := self.scrape(url)) :
                items.append(item)

        return items
