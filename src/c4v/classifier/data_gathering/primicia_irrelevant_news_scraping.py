"""
    Use this script to scrape data from primicia with non-relevant information, 
    and then store it in a csv file in its corresponding folder according to 
    this cookiecutter template:
    https://drivendata.github.io/cookiecutter-data-science/
"""
# Python imports
from typing import List
import datetime
import importlib.resources as resources

# Local imports
from c4v.scraper.crawler.crawlers.primicia_crawler import PrimiciaCrawler
from c4v.scraper.scraper import bulk_scrape

# Third Party imports
import pandas as pd
#TODO I have to refactor this code to use the manager class instead

# Crawl & scrape urls
LIMIT = 5000
crawler = PrimiciaCrawler.from_irrelevant()
urls = crawler.crawl_urls(up_to=5000)
data = bulk_scrape(urls)
del urls

# Convert to dataframe
df = pd.DataFrame(data)

# Set up datetime suffix
date_suffix = datetime.datetime.strftime(datetime.datetime.now(), "%Y%m%d%H%M%S")

with open(f"primicia_irrelevant_{date_suffix}.csv", "w+") as file:
    df.to_csv(file)
