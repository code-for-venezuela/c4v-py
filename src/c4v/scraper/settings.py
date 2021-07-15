"""
    This file manages multiple settings shared across the scraper,
    such as mappings from urls to scrapers
"""
from c4v.scraper.crawler.crawlers.base_crawler import BaseCrawler
from c4v.scraper.scrapers.base_scraper import BaseScraper
from c4v.scraper.scrapers.el_pitazo_scraper import ElPitazoScraper
from c4v.scraper.scrapers.primicia_scraper import PrimiciaScraper
from c4v.scraper.crawler.crawlers import primicia_crawler, el_pitazo_crawler
from c4v.scraper.utils import check_scrapers_consistency

import os
from typing import Type, List


INSTALLED_SCRAPERS: List[Type[BaseScraper]] = [ElPitazoScraper, PrimiciaScraper]

INSTALLED_CRAWLERS: List[Type[BaseCrawler]] = [primicia_crawler.PrimiciaCrawler, el_pitazo_crawler.ElPitazoCrawler]

SUPPORTED_DOMAINS  = [s.intended_domain for s in INSTALLED_SCRAPERS]

# Check for scraper consistency
check_scrapers_consistency(INSTALLED_SCRAPERS)

# Dict with information to map from domain to
# Spider
URL_TO_SCRAPER = {s.intended_domain: s for s in INSTALLED_SCRAPERS}

# root dir, so we can get resources from module directories
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

# Date format for this module:
DATE_FORMAT = "%Y-%m-%d %H:%M:%S.%f%z"
