"""
    Format columns of the client dataset in such a way that they match the ScrapedData format:

    url
    last_scraped
    title
    content
    author
    categories
    date
"""
# Python imports
import sys

# Third party
import pandas as pd
from pandas.core.indexing import is_nested_tuple

# Get command line argument
if len(sys.argv) < 2:
    print("Missing csv filename argument", file=sys.stderr)
    exit(1)

csv_file_name = sys.argv[1] # name of csv file
with open(csv_file_name) as file:
    df : pd.DataFrame = pd.read_csv(file)

# Rename columns
df.rename(
    {
        "text" : "content",
        "tipo_de_evento" : "label",
        "tags" : "categories" 
    }, 
    axis=1
    ,inplace=True)

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
df["last_scraped"] = None

# Write data @TODO write a platform independent version
filename = f"cleaned_{csv_file_name.split('/')[-1]}"
df.to_csv(filename)