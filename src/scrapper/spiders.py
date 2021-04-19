"""
    This file exports every spider or scrapper implementation for each site.
"""
# external imports
from re import S
import scrapy

#Project imports
import scrapper.utils as utils

#Python imports
from typing import List

class ElPitazoSpider(scrapy.Spider):
    """
        Spider to scrape ElPitazo data 
    """
    name = "el_pitazo"

    start_urls = []

    def parse(self, response):
        """
            Returns a dict like structure with the following 
            fields:
                + title
                + date
                + categories
                + body
                + author 
                + tags
        """

        # These are simple properties, just get its text with a valid 
        # selector
        title  = utils.get_element_text('.tdb-title-text', response) or ""
        date   = utils.get_element_text('.entry-date', response) or ""
        author = utils.get_element_text('.tdb-author-name', response) or ""

        body = self._get_body(response)

        tags = self._get_tags(response)

        # categories
        categories = response.css('.tdb-entry-category').getall()
        categories = list(map(utils.clean, categories))

        return {
            "title" : title,
            "date"  : date,
            "categories" : categories,
            "body" : body,
            "author" : author,
            "tags"   : tags
        }

    def _get_body(self, response) -> str:
        """
            Get article body as a single string
        """
        body = response.css("#bsf_rt_marker > p").getall()
        body = map(utils.clean, body)
        body = ''.join(body)

        return body

    def _get_tags(self, response) -> List[str]:
        """
            Try to get tags from document if available
        """
        tags = response.css(".tdb-tags > li > a").getall()
        tags = list(map(utils.clean, tags))
        return tags
