"""
    This data class represents data related to an url, such as last scraped data
    and url itself
"""

# Local imports 
from c4v.scraper.scraped_data_classes.scraped_data import ScrapedData

# Python imports 
from dataclasses import dataclass
from datetime import datetime

@dataclass()
class UrlData:
    """
        Represents an entry in a table holding data
        required to check if an url should be scraped 
    """

    url : str 
    last_scraped : datetime = None
    scraped_data : ScrapedData = None # notice that these two fields are nullable, it's such that you can base logic on it

