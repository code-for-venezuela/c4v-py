"""
    Use this script to process output data from the primicia_irrelevant_news_scraping.py script
    The file, in order to match the one used for trainning and provided by the client. Should provide the following collumns
    * label_relevance
    * label_service 
    * title
    * content
    * date
    * scraped_Date
    * author
    * url

    Expected arguments:
        csv_file_name : str = name of csv file with corresponding dataframe to use as basis
"""
import pandas as pd
import sys
import os
from pandas.core.frame import DataFrame

# Local imports
from c4v.scraper.scraped_data_classes.scraped_data import (
    RelevanceClassificationLabels,
    ServiceClassificationLabels,
)

# Get command line argument
if len(sys.argv) < 2:
    print("Missing csv filename argument", file=sys.stderr)
    exit(1)

csv_file_name = sys.argv[1]  # name of csv file
try:
    df: DataFrame = pd.read_csv(csv_file_name)
except Exception as e:
    print(
        f"[Error] Could not write to provided file: {csv_file_name}.\n\tError: {e}",
        file=sys.stderr,
    )
    exit(1)

# Add label column as irrelevant for every row
print("Adding irrelevant label to every row...")
df["label_relevance"] = RelevanceClassificationLabels.IRRELEVANTE.value
df["label_service"] = ServiceClassificationLabels.NO_SERVICIO.value

# Remove empty content
print("Removing rows with empty values...")
df.dropna(inplace=True, axis=0)
df["content"].drop(
    df[(df.content.str.len() < 1) | (df.title.str.len() < 1)].index, inplace=True
)


# Remove extra linejumps
print("Removing extra line jumps from article bodies...")


def strip_extra_linejumps(s: str) -> str:
    return "\n".join([line for line in s.splitlines() if line.strip() != ""])


df["content"] = df["content"].map(strip_extra_linejumps)
# Remove duplicates if any
print("Removing duplicates rows if any...")
df.drop_duplicates(inplace=True, subset="url")

# Remove unnecesary columns
columns = [
    "label_relevance",
    "label_service",
    "content",
    "title",
    "author",
    "date",
    "last_scraped",
    "categories",
    "url",
]
columns_to_remove = [c for c in df.columns if c not in columns]
print(f"Removing the following columns: {columns_to_remove}")
df.drop(columns_to_remove, inplace=True, axis="columns")
print("Cleaned Dataset Shape: ")
print(df)

# Save resulting file
filename = "cleaned_" + os.path.basename(csv_file_name)
print(f"Saving cleaned data to: {filename}")
df.to_csv(filename, index=False)
