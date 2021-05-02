# External imports
import scrapy

# Python imports
from typing import Dict, Any


class SpiderPrimicia(scrapy.Spider):
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

    def parse(self, response, **kwargs) -> Dict[str, Any]:
        """
        Returns a dict like structure with the following
        fields:
            + title
            + body
            + date
            + author
            + categories
            + tags
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

        return {
            "title": title,
            "body": body,
            "date": date,
            "author": author,
            "categories": categories,
            "tags": tags,
        }

    def _get_body(self, response, selector) -> str:
        """
        Get article body as a single unicode string
        """
        body = response.css(selector).getall()
        body = "\n".join(body)

        return body.strip()
