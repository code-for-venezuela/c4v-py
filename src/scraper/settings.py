"""
    This file manages multiple settings shared across the scrapper,
    such as mappings from urls to spiders
"""
from scraper.scrapers.el_pitazo_scraper import ElPitazoScraper

# Dict with information to map from domain to
# Spider
URL_TO_SCRAPER = {
    "elpitazo.net": ElPitazoScraper,
}
