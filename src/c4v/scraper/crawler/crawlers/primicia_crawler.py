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
    
    @staticmethod
    def check_sitemap_url(url: str) -> bool:
        return url.startswith("https://primicia.com.ve/post-sitemap")

    @staticmethod
    def check_page_url(url: str) -> bool:
        primicia = "https://primicia.com.ve/"
        return url.startswith(primicia) and len(url) > len(primicia)
