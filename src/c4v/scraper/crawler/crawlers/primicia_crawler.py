"""
    Class for scraping primicia data
"""
from c4v.scraper.crawler.crawlers.base_crawler import BaseCrawler


class PrimiciaCrawler(BaseCrawler):
    """
        Class to crawl Primicia Urls
    """

    start_sitemap_url = "https://primicia.com.ve/sitemap_index.xml"
    name = "primicia"

    # URL patterns for non relevant urls (news that have nothing to do with public services)
    IRRELEVANT_URLS = [
        ".*primicia.com.ve/mundo/.*",
        ".*primicia.com.ve/deportes/.*",
        ".*primicia.com.ve/placeres/.*",
        ".*primicia.com.ve/mas/virales/.*",
        ".*primicia.com.ve/mas/ciencia-y-tecnologia/.*",
        ".*primicia.com.ve/sucesos/.*",
        ".*primicia.com.ve/politica/.*",
    ]

    @staticmethod
    def should_crawl(url: str) -> bool:
        # Sitemaps about posts will start with this prefix
        return url.startswith("https://primicia.com.ve/post-sitemap")

    @staticmethod
    def should_scrape(url: str) -> bool:
        # Checks if provided url starts with base site
        # and if its length is creater to base site (so we avoid crawling main page
        # by accident)
        primicia = "https://primicia.com.ve/"
        return url.startswith(primicia) and len(url) > len(primicia)
