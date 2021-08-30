"""
    Use this script to scrape data from primicia with non-relevant information, 
    and then store it in a csv file in its corresponding folder according to 
    this cookiecutter template:
    https://drivendata.github.io/cookiecutter-data-science/
"""
# Python imports
import datetime

# Local imports
from c4v.scraper.crawler.crawlers.primicia_crawler import PrimiciaCrawler
from c4v.scraper.scraper import bulk_scrape

# Third Party imports
import pandas as pd
#TODO I have to refactor this code to use the manager class instead

# Take time:
start = datetime.datetime.now()

# Crawl & scrape urls
LIMIT = 10000
print(f"Scraping up to {LIMIT} urls from primicia")

crawler = PrimiciaCrawler.from_irrelevant()
urls = crawler.crawl_urls(up_to=LIMIT)

print(f"URLs scrawled, gathered: {len(urls)}. Scraping new urls...")

data = bulk_scrape(urls)
del urls

print(f"URLs scraped: {len(data)}. Converting to dataframe and saving to csv...")

# Convert to dataframe
df = pd.DataFrame(data)

# Set up datetime suffix
date_suffix = datetime.datetime.strftime(datetime.datetime.now(), "%Y%m%d%H%M%S")
filename = f"primicia_irrelevant_{date_suffix}.csv" 

with open(filename, "w+") as file:
    df.to_csv(file)

print(f"Data saved to {filename}")

end = datetime.datetime.now()

print(f"Total Time: {(end - start).total_seconds()}s")