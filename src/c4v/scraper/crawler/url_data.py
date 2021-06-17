"""
    This data class represents data related to an url, such as last scraped data
    and url itself
"""

# Local imports 
from typing import Dict, Any
from c4v.scraper.scraped_data_classes.scraped_data import ScrapedData

# Python imports 
import dataclasses
from datetime import datetime
import json 

@dataclasses.dataclass
class UrlData:
    """
        Represents an entry in a table holding data
        required to check if an url should be scraped 
    """

    url : str 
    last_scraped : datetime = None
    scraped_data : ScrapedData = None # notice that these two fields are nullable, it's such that you can base logic on it


class UrlDataEncoder(json.JSONEncoder):
    """
        Encoder to turn this file into json format
    """
    def default(self, obj: UrlData) -> Dict[str, Any]:
        if isinstance(obj, UrlData):
            return dataclasses.asdict(obj)
        
        return json.JSONEncoder.default(self, obj)