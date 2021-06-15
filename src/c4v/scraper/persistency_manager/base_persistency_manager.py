"""
    Base Persistency Manager: We need a way to handle persistency, 
    as simple files or maybe with a more complex scheme, such as databases
    or http requests. This base class provides a contract that
    every persistency manager should match in order to be used with our app.

    Notice that 
"""

# Python imports
from typing import List

# Local imports
from c4v.scraper.crawler.url_data import UrlData

class BasePersistencyManager:
    """
        Base class to provide support for persistency management
    """

    def get_known_data_for( self, 
                            urls : List[str], 
                            skip_body : bool = False) -> List[UrlData]:
        """
        Retrieve data known for a given list of urls. Retrieved 
        data should be an instance of UrlData and its field should match:
            -  url field is always present
            -  Every Given url in the input should be present in the output list 
            -  last scraped date field should be null when this url is not scraped yet
            -  if skip_body == True, then all UrlData instances will have its scraped_data field as null

        Parameters:
            - urls : [str] = list of urls 
            - skip_body : bool = If should skip scraped_data field of urls (may improve performance for some
                                 operations where the body is not needed)
        Return:
            List of url data matching provided urls
        """
        raise NotImplementedError("Implement get_known_data_for abstract method")


    def save(self, url_data : List[UrlData]):
        """
            Save provided data to local storage
            Parameters:
                - data : [UrlData] = data to be saved
        """
        raise NotImplementedError("Implement save abstract method")

    def delete(self, urls : List[str]):
        """
            Delete provided urls from persistent storage
            Parameters:
                - urls : [str] = Urls to be deteled
        """
        raise NotImplementedError("Implement delete abstract method")
