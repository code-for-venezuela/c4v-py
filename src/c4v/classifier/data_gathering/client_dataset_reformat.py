"""
    Format columns of the client dataset in such a way that they match the ScrapedData format, plus a label field:

    url
    last_scraped
    title
    content
    author
    categories
    date
    source
    label_service
    label_relevance

    # Parameters 
    - client_dataset_file : `str` = filename of the client dataset

    --------------------------------------------------------
    Note that this script won't check for duplicate labels:
    
    key  | a | b | c | label          Key | a | d | label 
    ------------------------          --------------------
    1    | _ | _ | _ | l1             1   | _ | _ | l1
    1    | _ | _ | _ | l2       --->  1   | _ | _ | l2
    2    | _ | _ | _ | l3             2   | _ | _ | l3
    2    | _ | _ | _ | l3
"""
# Local imports 
from c4v.scraper.scraped_data_classes.scraped_data import Sources, RelevanceClassificationLabels

# Python imports
import sys
import os
import datetime
import pytz

# Third party
import pandas as pd

# Get command line argument
if len(sys.argv) < 2:
    print("Missing csv filename argument", file=sys.stderr)
    exit(1)

csv_file_name = sys.argv[1]  # name of csv file
with open(csv_file_name) as file:
    df: pd.DataFrame = pd.read_csv(file)

# Rename columns
print("Columns are: ", df.columns)

df.rename(
    {
        "text": "content",
        "tipo_de_evento": "label",
        "tags": "categories",
        "Link de la noticia ": "url",
        "tipo de evento": "label_relevance",
        "servicio_resumido" : "label_service"
    },
    axis="columns",
    inplace=True,
)

# Valid columns
columns = ["url", "title", "content", "author", "categories", "date", "label_relevance", "label_service"]

# Remove irrelevant columns
columns_to_remove = [col for col in df.columns if col not in columns]
print(f"Removing columns: {columns_to_remove}")

df.drop(columns_to_remove, inplace=True, axis=1)
print(f"Columns: {df.columns}")

# Adding source column
source_col_val = Sources.CLIENT.value
print(f"Adding 'source' column with value: {source_col_val}")
df['source'] = source_col_val

# Update label relevance to be relevant for every instance
df["label_relevance"] = RelevanceClassificationLabels.DENUNCIA_FALTA_DEL_SERVICIO.value

# Complete missing columns:
df["last_scraped"] = datetime.datetime.now(tz=pytz.utc)

filename = f"cleaned_{os.path.basename(csv_file_name)}"
print(f"Saving file to: {filename}")
df.to_csv(filename, index=False)
