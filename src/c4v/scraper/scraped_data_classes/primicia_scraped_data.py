"""
    Base class for Primicia sites scraped data. Includes:
        + body
        + title
        + author
        + tags
        + categories
"""
# Local imports
from c4v.scraper.scraped_data_classes.base_scraped_data import BaseDataFormat
from c4v.scraper.scraped_data_classes.scraped_data import ScrapedData

# Python imports
from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class PrimiciaData(BaseDataFormat):
    """
        Scraped data expected for ElPitazo news articles
    """

    tags: List[str]
    categories: List[str]
    title: str
    author: str
    date: str
    body: str

    def to_scraped_data(self) -> ScrapedData:
        return ScrapedData(
            author=self.author,
            date=self.date,
            title=self.title,
            categories=self.tags + self.categories,
            content=self.body,
            url=self.url,
            last_scraped=self.last_scraped
        )
