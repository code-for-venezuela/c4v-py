"""
    Misc. helper functions.
"""
# External imports
from bs4 import BeautifulSoup
from tabulate import tabulate
from c4v.scraper.scraped_data_classes.scraped_data import ScrapedData

# local imports
from c4v.scraper.scrapers.base_scraper import BaseScraper

# Python imports
import re
import pytz
from datetime import datetime
from urllib.parse import urlparse
from typing import List, Type


def strip_http_tags(element: str) -> str:
    """
        Get text from a html formated string
        Parameters:
            + element - str : html formated string to be cleaned
        Return:
            String cleaned from html tags, leaving text only
    """

    # use BeautifulSoup to clean string
    soup = BeautifulSoup(element, "lxml")
    return soup.get_text()


def get_element_text(selector: str, response) -> str:
    """
        Return cleaned text from an element selected by "selector".
        May return None if element was not found
        Parameters: 
            + selector - str : valid css selector used to select html element 
            + response : response as passed to parse in Spider class
        Return:
            Selected element's content as a cleaned string, or None if such 
            element is not present
    """
    # try to select element from document
    value = response.css(selector)

    # in case nothing found, return None
    if not value:
        return None

    # clean from html tags
    soup = BeautifulSoup(value.get(), "lxml")

    return soup.get_text()


def valid_url(url: str) -> bool:
    """
        Check that the given url is actually a valid url
    """
    url_matcher = re.compile(
        r"^(?:http|ftp)s?://"  # http:// or https://
        r"(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+(?:[A-Z]{2,6}\.?|[A-Z0-9-]{2,}\.?)|"  # domain...
        r"localhost|"  # localhost...
        r"\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})"  # ...or ip
        r"(?::\d+)?"  # optional port
        r"(?:/?|[/?]\S+)$",
        re.IGNORECASE,
    )

    return re.match(url_matcher, url) is not None


def get_domain_from_url(url: str) -> str:
    """
        Get domain name from  a valid url 
        Parameters:
            + url : str = url to get domain from
        Return:
            domain name for the given url
    """

    return urlparse(url).netloc


def check_scrapers_consistency(scrapers: List[Type[BaseScraper]]):
    """
        Check consistency for a scraper list
    """
    for scraper in scrapers:
        check_scraper_consistency(scraper)


def check_scraper_consistency(scraper: Type[BaseScraper]):
    """
        Check consistency for a scraper. This function checks that:
            + Scraper provides intended domain
    """
    assert scraper.intended_domain != None and isinstance(
        scraper.intended_domain, str
    ), f"Scraper {scraper} does not provide intended_domain"


def get_datetime_now() -> datetime:
    """
        Return datetime formated properly
    """
    return datetime.now(tz=pytz.UTC)


def data_list_to_table_str(data_list: List[ScrapedData], max_cell_len: int = 50) -> str:
    """
        Return a string representation of a given data list in a tabular format
        Parameters:
            + data_list : [ScrapedData] = list of data instances to list in table
    """

    cols = [k for k in ScrapedData.__dataclass_fields__.keys()]
    cols.sort()
    headers = ["row"] + cols

    truncate = lambda s: s if len(s) <= max_cell_len else s[:max_cell_len] + "..."

    rows = [
        [i + 1] + [truncate(str(data.__getattribute__(attr))) for attr in cols]
        for (i, data) in enumerate(data_list)
    ]

    return tabulate(rows, headers=headers)
