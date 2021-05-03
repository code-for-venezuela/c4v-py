# Internal imports
import c4v.scraper.utils as utils

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

    def parse(self, response) -> Dict[str, Any]:
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
        title = response.css(".tdb-title-text ::text").get() or ""
        date =  response.css(".entry-date ::text").get() or ""
        author = response.css(".tdb-author-name ::text").get() or ""

        body = self._get_body(response)

        tags = self._get_tags(response)

        # categories
        categories = response.css(".tdb-entry-category")
        categories = list(map(lambda c: c.css("::text").get(), categories))

        return {
            "title": title,
            "date": date,
            "categories": categories,
            "body": body,
            "author": author,
            "tags": tags,
        }

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
        tags = response.css(".tdb-tags > li > a")
        tags = list(map(lambda t: t.css("::text").get(), tags))
        return tags
