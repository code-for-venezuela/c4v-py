from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class ScrapedData:
    """
        Data class that maps directly to a database schema. Every scraper
        should return its own instance of this class
    """

    title: str = None
    content: str = None
    author: str = None
    categories: List[str] = None
    date: str = None
