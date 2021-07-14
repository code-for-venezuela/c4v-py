# Internal imports
import c4v.scraper.utils as utils
from c4v.scraper.scraped_data_classes.elpitazo_scraped_data import ElPitazoData

# External imports
import scrapy
from scrapy.http import Response

# Python imports
from typing     import List, Dict, Any
from datetime   import datetime
import pytz

class ElPitazoSpider(scrapy.Spider):
    """
        Spider to scrape ElPitazo data 
    """

    name = "el_pitazo"

    start_urls = []

    def parse(self, response: Response) -> ElPitazoData:
        """
            Returns a data object describing data scrapable for this 
            scraper

            Parameters:
                + response : Response = scrapy response object
            
            Return:
                An ElPitazoData instance 
        """

        # These are simple properties, just get its text with a valid
        # selector
        title = response.css(".tdb-title-text ::text").get() or ""
        date = response.css(".entry-date ::text").get() or ""
        author = response.css(".tdb-author-name ::text").get() or ""

        body = self._get_body(response)

        tags = self._get_tags(response)

        # categories
        categories = response.css(".tdb-entry-category ::text").getall()

        return ElPitazoData(
            body=body,
            tags=tags,
            categories=categories,
            title=title,
            author=author,
            date=date,
            url=response.url,
            last_scraped = utils.get_datetime_now()
        )

    def _get_body(self, response: Response) -> str:
        """
            Get article body as a single string
        """
        body = response.css("#bsf_rt_marker > p").getall()
        body = filter(lambda p: p.startswith("<p>") and p.endswith("</p>"), body)
        body = map(utils.strip_http_tags, body)

        body = "\n".join(body)

        return body.strip()

    def _get_tags(self, response: Response) -> List[str]:
        """
            Try to get tags from document if available
        """
        tags = response.css(".tdb-tags > li > a ::text").getall()
        return tags
