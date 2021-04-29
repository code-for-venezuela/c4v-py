"""
    Base class for news sites scraped data. Includes:
        + body
        + title
        + author
        + tags
        + categories
"""
# Local imports
from c4v.scraper.scraped_data_classes.base_scraped_data import BaseScrapedData

# Python imports
from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class ElPitazoData(BaseScrapedData):
    """
        Scraped data expected for ElPitazo news articles
    """

    tags: List[str]
    categories: List[str]
    title: str
    author: str
    date: str
