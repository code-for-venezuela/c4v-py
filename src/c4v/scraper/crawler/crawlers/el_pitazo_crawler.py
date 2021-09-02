"""
    Crawler for ElPitazo site. 
"""

# Local imports
from c4v.scraper.crawler.crawlers import base_crawler

# Third party imports


# Python imports
from typing import List


class ElPitazoCrawler(base_crawler.BaseCrawler):
    """
        Class to crawl El Pitazo urls 
    """

    start_sitemap_url = "https://elpitazo.net/sitemap.xml"
    name = "el_pitazo"

    @staticmethod
    def should_crawl(url: str) -> bool:
        return url.startswith("https://elpitazo.net/sitemap-pt-post-")
