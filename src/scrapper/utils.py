"""
    Misc. helper functions.
"""
from bs4 import BeautifulSoup

def clean(element : str) -> str:
    """
        Get text from a html formated string
        Parameters:
            + element - str : html formated string to be cleaned
        Return:
            String cleaned from html tags, leaving text only
    """

    # use BeautifulSoup to clean string
    soup = BeautifulSoup(element, 'lxml')
    return soup.get_text()

def get_element_text(selector : str, response) -> str:
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
    soup = BeautifulSoup(value.get(), 'html')

    return soup.get_text()