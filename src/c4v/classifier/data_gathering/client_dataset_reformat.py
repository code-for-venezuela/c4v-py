"""
    Format columns of the client dataset in such a way that they match the ScrapedData format, plus a label field:

    url
    last_scraped
    title
    content
    author
    categories
    date
    label
"""
# Python imports
import sys
import os
import datetime
import pytz

# Third party
import pandas as pd
from pandas.core.indexing import is_nested_tuple

# Get command line argument
if len(sys.argv) < 2:
    print("Missing csv filename argument", file=sys.stderr)
    exit(1)

csv_file_name = sys.argv[1]  # name of csv file
with open(csv_file_name) as file:
    df: pd.DataFrame = pd.read_csv(file)

# Rename columns
print("Columnds are: ", df.columns)

df.rename(
    {
        "text": "content",
        "tipo_de_evento": "label",
        "tags": "categories",
        "Link de la noticia ": "url",
        "tipo de evento": "label",
    },
    axis=1,
    inplace=True,
)

# Valid columns
columns = ["url", "title", "content", "author", "categories", "date", "label"]

# Remove irrelevant columns
columns_to_remove = [col for col in df.columns if col not in columns]
print(f"Removing columns: {columns_to_remove}")

df.drop(columns_to_remove, inplace=True, axis=1)
print(f"Columns: {df.columns}")
old_df = df.copy()
url_to_labels_df = old_df.groupby("url")

# Remove duplicates
df.drop_duplicates(inplace=True, ignore_index=True, subset=["url"])

# Try to edit label field
for (url, sub_df) in url_to_labels_df:
    labels = list(set(sub_df.label))
    df.loc[df.url == url, "label"] = [labels]

# Complete missing columns:
df["last_scraped"] = datetime.datetime.now(tz=pytz.utc)

filename = f"cleaned_{os.path.basename(csv_file_name)}"
df.to_csv(filename)
