"""
    Base Persistency Manager: We need a way to handle persistency, 
    as simple files or maybe with a more complex scheme, such as databases
    or http requests. This base class provides a contract that
    every persistency manager should match in order to be used with our app.
"""

# Python imports
from typing import Iterator, List

# Local imports
from c4v.scraper.scraped_data_classes.scraped_data import ScrapedData


class BasePersistencyManager:
    """
        Base class to provide support for persistency management
    """

    def get_all(
        self, limit: int = -1, scraped: bool = None, order_by: List[str] = None
    ) -> Iterator[ScrapedData]:
        """
            Return an iterator over the set of stored instances
            Parameters:
                + limit : int = Max amount of elements to retrieve
                + scraped : bool = True if retrieved data should be scraped, false if it shouldn't, None if not relevant
                + order_by : str = (optional) names of the fields to use for sorting, first char should be order, - for descending, + for ascending, 
                                  following chars in each string should be a valid name of a field in the ScrapedData dataclass.
                                  If no provided, no order is ensured
            Return:
                Iterator of stored ScrapedData instances
        """
        raise NotImplementedError("Implement filter_scraped_urls abstract function")

    def filter_known_urls(self, urls: List[str]) -> List[str]:
        """
            Filter out urls that are already known to the database, leaving only 
            the ones that are new 
        """
        raise NotImplementedError("Implement filter_known_urls abstract function")

    def filter_scraped_urls(self, urls: List[str]) -> List[str]:
        """
            Filter out urls whose data is already known, leaving only the ones to be scraped
            for first time
            Parameters: 
                urls : [str] = List of urls to filter 

            Return:
                A list of urls such that none of them has been scraped 
        """
        raise NotImplementedError("Implement filter_scraped_urls abstract function")

    def was_scraped(self, url: str) -> bool:
        """
            Tells if a given url is already scraped (it's related data is already know)
            Parameters:
                url : str = url to check if it was already scraped
            Return:
                If the given url's related data is already known
                
        """
        raise NotImplementedError("Implement was_scraped abstract function")

    def save(self, url_data: List[ScrapedData]):
        """
            Save provided data to local storage. 
            If some some urls are already in local storage, update them with provided new data.
            If not, delete them.
            Parameters:
                - data : [ScrapedData] = data to be saved
        """
        raise NotImplementedError("Implement save abstract method")

    def delete(self, urls: List[str]):
        """
            Delete provided urls from persistent storage
            Parameters:
                - urls : [str] = Urls to be deteled
        """
        raise NotImplementedError("Implement delete abstract method")
