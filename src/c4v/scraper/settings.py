"""
    This file manages multiple settings shared across the scraper,
    such as mappings from urls to scrapers
"""
from c4v.scraper.scrapers.el_pitazo_scraper import ElPitazoScraper
from c4v.scraper.utils import check_scrapers_consistency

import os

INSTALLED_SCRAPERS = [ElPitazoScraper]

# Check for scraper consistency
check_scrapers_consistency(INSTALLED_SCRAPERS)

# Dict with information to map from domain to
# Spider
URL_TO_SCRAPER = {s.intended_domain: s for s in INSTALLED_SCRAPERS}


# root dir, so we can get resources from module directories
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
