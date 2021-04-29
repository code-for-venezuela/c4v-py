"""
    Base class for scraped data
"""

# Python imports:
from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)  # Readonly attributes
class BaseScrapedData:
    """
        Base class for every scraper output class
    """

    body: str
