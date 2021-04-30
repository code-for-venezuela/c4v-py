from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
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

    title: str = None
    content: str = None
    author: str = None
    categories: List[str] = None
    date: str = None
