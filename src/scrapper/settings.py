"""
    This file manages multiple settings shared across the scrapper,
    such as mappings from urls to spiders
"""
import scrapper.spiders as spiders

# Dict with information to map from domain to 
# Spider
URL_TO_SPIDERS = {
    "elpitazo.net" : spiders.ElPitazoSpider,
}

# Settings passed to the crawler
CRAWLER_SETTINGS = {"LOG_ENABLED" : True} 