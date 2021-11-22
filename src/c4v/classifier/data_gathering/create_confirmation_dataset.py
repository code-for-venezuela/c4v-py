"""
    Creates a confirmation dataset, using a list of well-known urls for news that have problems.
    Expected fields for such csv:
        + url
        + label
    Arguments:
        valid_dataset_filename : str = path to file to load as a dataframe containing only urls to news with positive labels
        train_dataset_filename : str = path to training dataset, used to know which urls won't be used for confirmation dataset
        db_name : str = (optional) filename of the db used to retrieve non-relevant urls, by default using the local sqlite db
        n_irrelevant_rows : int = (optional) amount of rows to be used for irrelevant news
"""
# Local imports
from typing import List
from c4v.config import settings
from c4v.microscope.manager import Manager
from c4v.scraper.crawler.crawlers.primicia_crawler import PrimiciaCrawler

# Third party imports
import pandas as pd
import sys

# Ammount of arguments for this script
_N_ARGS = 2

# Sanity check: Enough arguments
if len(sys.argv) < (_N_ARGS + 1):
    print("[ERROR] Not enough arguments", file=sys.stderr)

# Read arguments from command line
# Read positive labels dataset
valid_dataset_filename: str = sys.argv[1]
print(f"Reading positive labels dataset: {valid_dataset_filename}...")
valid_dataset_df = pd.read_csv(valid_dataset_filename)
valid_dataset_df.dropna(axis=0, inplace=True, subset=["url"])

# Read training dataset
train_dataset_filename: str = sys.argv[2]
print(f"Reading training  dataset: {train_dataset_filename}...")
train_dataset_df = pd.read_csv(train_dataset_filename)

# Read sqlite db argument
db_name = sys.argv[3] if len(sys.argv) > (_N_ARGS + 1) else settings.local_sqlite_db

# Read how much irrelevant rows to use
n_irrelevant_rows: int = int(sys.argv[3]) if len(sys.argv) > (_N_ARGS + 1) else 10000
print(
    f"Will try to retrieve {n_irrelevant_rows} irrelevant rows from database {db_name}..."
)

# Set up manager
m: Manager = Manager.from_local_sqlite_db(db_name)

# -- Collecting urls to scrape -----------------------------------

# Get urls to scrape as problems
positive_urls_to_scrape, non_scrapable = m.split_non_scrapable(
    list(valid_dataset_df["url"])
)

if non_scrapable:  # warn about non-scrapable urls
    print(
        f"[WARNING] {len(non_scrapable)} urls from {len(valid_dataset_df)} won't be scraped, as they're not scrapable",
        file=sys.stderr,
    )
    del non_scrapable

# Get data not in training dataset
not_valid_urls = set(train_dataset_df["url"]).union(positive_urls_to_scrape)
del train_dataset_df

print(f"Crawling irrelevant data...")
# Crawl data for primicia
primicia_crawler = PrimiciaCrawler.from_irrelevant()
# crawled urls
irrelevant_urls: List[str] = []


def collect(crawled_urls: List[str]):
    irrelevant_urls.extend(url for url in crawled_urls if url not in not_valid_urls)
    print(f"Collected items: {len(irrelevant_urls)} / {n_irrelevant_rows}")


def should_stop() -> bool:
    return len(irrelevant_urls) >= n_irrelevant_rows


irrelevant_urls = irrelevant_urls[:n_irrelevant_rows]

# Crawl items
primicia_crawler.crawl_and_process_urls(
    post_process_data=collect, should_stop=should_stop
)

print(f"Collected irrelevant articles: {len(irrelevant_urls)}")

# Scrape crawled items

# -- Scraping urls -------------------------------
# Create positive dataframe
print(f"Scraping collected items...")
scraped_data = m.get_bulk_data_for(
    positive_urls_to_scrape + irrelevant_urls
)  # get data
scraped_positive_data_df = pd.DataFrame(
    x for x in scraped_data if x.url in positive_urls_to_scrape
)  # turn into dataset
# remove unnecesary rows:
valid_dataset_df.drop(
    [c for c in valid_dataset_df.columns if c != "label" and c != "url"],
    inplace=True,
    axis=1,
)
scraped_positive_data_df = pd.merge(  # join dataframes to get corresponding labels
    scraped_positive_data_df, valid_dataset_df, "inner", on="url"
)

# Create Irrelevant dataframe
irrelevant_scraped_df = pd.DataFrame(
    x for x in scraped_data if x.url in irrelevant_urls
)

# Remove irrelevant cols
irrelevant_scraped_df.drop(
    [
        c
        for c in irrelevant_scraped_df.columns
        if c not in scraped_positive_data_df.columns
    ],
    inplace=True,
    axis=1,
)
irrelevant_scraped_df["label"] = [["IRRELEVANTE"]] * len(irrelevant_scraped_df)

full_df = pd.concat(
    [scraped_positive_data_df, irrelevant_scraped_df], ignore_index=True
)
full_df = full_df.sample(frac=1.0)
print(full_df)
print("Creating final dataset...")

filename = "confirmation_dataset.csv"
print(f"Saving file to: {filename}")
full_df.to_csv(filename, index=False)
