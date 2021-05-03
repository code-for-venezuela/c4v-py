"""
    This file manages multiple settings shared across the scraper,
    such as mappings from urls to scrapers
"""
from c4v.scraper.scrapers.el_pitazo_scraper import ElPitazoScraper
from c4v.scraper.scrapers.scraper_primicia import ScraperPrimicia
import os


# Dict with information to map from domain to
# Spider
URL_TO_SCRAPER = {
    "elpitazo.net": ElPitazoScraper,
    "primicia.com.ve": ScraperPrimicia
}


# root dir, so we can get resources from module directories
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

# todo: find a better way to get c4v-py abs path.  Root library directory.
C4V_ROOT_DIR = "/".join(ROOT_DIR.split("/")[:-3])
