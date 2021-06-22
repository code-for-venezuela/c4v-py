# Python imports
from dataclasses    import dataclass, asdict
from typing         import List, Dict, Any
from datetime       import datetime
import json

@dataclass()
class ScrapedData:
    """
        This is a general data format class, 
        every data format for other scrapers could have 
        additional fields according to its needs and
        scrapable data, but then they should be able to 
        convert themselves into this format, possibly leaving a 
        few fields as None. Thus, we can be able to 
        easily map from a scrapers's output to a database 
        scheme
    """

    url: str
    last_scraped: datetime = None
    title: str = None
    content: str = None
    author: str = None
    categories: List[str] = None
    date: str = None


class ScrapedDataEncoder(json.JSONEncoder):
    """
        Encoder to turn this file into json format
    """
    def default(self, obj: ScrapedData) -> Dict[str, Any]:
        if isinstance(obj, ScrapedData):
            return asdict(obj)
        elif isinstance(obj, datetime):

            # Local imports
            from c4v.scraper.settings   import DATE_FORMAT
            return datetime.strftime(obj, DATE_FORMAT)

        return json.JSONEncoder.default(self, obj)