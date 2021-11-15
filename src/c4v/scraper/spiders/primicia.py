# External imports
import scrapy
from scrapy.http import Response

# Local imports
from .. import utils 
from c4v.scraper.scraped_data_classes.primicia_scraped_data import PrimiciaData


class PrimiciaSpider(scrapy.Spider):
    """
    Spider to scrape Primicia data
    """

    name = "primicia"

    start_urls = []

    # Selectors of Primicia
    TITLE_SELECTOR = "h1.title-single ::text"
    BODY_SELECTOR = ".credit-photoleyends + div ::text"
    DATE_SELECTOR = "#namesingle + div small ::text"
    AUTHOR_SELECTOR = "#namesingle a.author-link ::text"
    TAGS_SELECTOR = ".badge.badge-pill.badge-primicia ::text"
    CATEGORIES_SELECTOR = None

    def parse(self, response: Response, **kwargs) -> PrimiciaData:
        """
        Return a PrimiciaData instance, describing data possible scraped for this page
        Parameters:
            + response : Response = Scrapy response object
        Return:
            Data scraped for this page
        """

        title = response.css(self.TITLE_SELECTOR).get() or ""
        body = self._get_body(response, self.BODY_SELECTOR)
        date = response.css(self.DATE_SELECTOR).get() or ""
        author = response.css(self.AUTHOR_SELECTOR).get() or ""
        tags = response.css(self.TAGS_SELECTOR).getall() if self.TAGS_SELECTOR else []
        categories = (
            response.css(self.CATEGORIES_SELECTOR).getall()
            if self.CATEGORIES_SELECTOR
            else []
        )

        return PrimiciaData(
            tags=tags,
            categories=categories,
            title=title,
            author=author,
            date=date,
            body=body,
            url=response.url,
            last_scraped=utils.get_datetime_now(),
        )

    def _get_body(self, response: Response, selector) -> str:
        """
        Get article body as a single unicode string
        """
        body = response.css(selector).getall()
        body = "\n".join(body)

        return body.strip()
