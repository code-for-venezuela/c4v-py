"""
    Base class for scraped data
"""

# Local imports
from .scraped_data import ScrapedData

# Python imports:
from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)  # Readonly attributes
class BaseDataFormat:
    """
        Base class for every scraper output class
    """

    def to_scraped_data(self) -> ScrapedData:
        """
            This function transforms from a data format 
            to our standard ScrapedData format
        """
        raise NotImplementedError(
            "Every BaseScrapedData sub class should implement to_scraped_data method"
        )
