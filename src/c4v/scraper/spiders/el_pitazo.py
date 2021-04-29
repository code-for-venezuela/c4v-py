# Internal imports
import c4v.scraper.utils as utils
from c4v.scraper.scraped_data_classes.elpitazo_scraped_data import ElPitazoData

# External imports
import scrapy

# Python imports
from typing import List, Dict, Any


class ElPitazoSpider(scrapy.Spider):
    """
        Spider to scrape ElPitazo data 
    """

    name = "el_pitazo"

    start_urls = []

    def parse(self, response) -> ElPitazoData:
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
        title = utils.get_element_text(".tdb-title-text", response) or ""
        date = utils.get_element_text(".entry-date", response) or ""
        author = utils.get_element_text(".tdb-author-name", response) or ""

        body = self._get_body(response)

        tags = self._get_tags(response)

        # categories
        categories = response.css(".tdb-entry-category").getall()
        categories = list(map(utils.strip_http_tags, categories))

        return ElPitazoData(
            body=body,
            tags=tags,
            categories=categories,
            title=title,
            author=author,
            date=date,
        )

    def _get_body(self, response) -> str:
        """
            Get article body as a single string
        """
        body = response.css("#bsf_rt_marker > p").getall()
        body = filter(lambda p: p.startswith("<p>") and p.endswith("</p>"), body)
        body = map(utils.strip_http_tags, body)

        body = "\n".join(body)

        return body.strip()

    def _get_tags(self, response) -> List[str]:
        """
            Try to get tags from document if available
        """
        tags = response.css(".tdb-tags > li > a").getall()
        tags = list(map(utils.strip_http_tags, tags))
        return tags
