"""
    Use this script to scrape data from primicia with non-relevant information, 
    and then store it in a csv file in the current working directory

    Script Arguments:
        limit (position 1) : max ammount of urls to crawl
"""
# Python imports
import datetime
import sys

# Local imports
from c4v.scraper.crawler.crawlers.primicia_crawler import PrimiciaCrawler
from c4v.scraper.scraper import bulk_scrape

# Third Party imports
import pandas as pd

# Take time:
start = datetime.datetime.now()

# Set up max ammount of urls to crawl:
if len(sys.argv) < 2:
    LIMIT = 10000
    print(f"[WARNING] No Limit provided, defaulting to {LIMIT}")
else:
    try:
        LIMIT = int(sys.argv[1])
    except Exception as e:
        print(
            f"Invalid limit argument. Are you sure it is a valid number?\n\tError: {e}",
            file=sys.stderr,
        )
        exit(1)

# Crawl & scrape urls
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
