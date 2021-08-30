"""
    Use this script to process output data from the primicia_irrelevant_news_scraping.py script
    The file, in order to match the one used for trainning and provided by the client, should provide the following collumns
    * label
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
import datetime

from pandas.core.frame import DataFrame

# Get command line argument
if len(sys.argv) < 2:
    print("Missing csv filename argument", file=sys.stderr)
    exit(1)

csv_file_name = sys.argv[1] # name of csv file
with open(csv_file_name) as file:
    df : DataFrame = pd.read_csv(file)

# Remove empty content
df.dropna(inplace=True, axis=0)
df['content'].drop(df[(df.content.str.len() < 1) | (df.title.str.len() < 1) ].index, inplace=True)

# Add label column as irrelevant for every row
df['label'] = "IRRELEVANTE"

# Remove extra linejumps
def strip_extra_linejumps(s : str) -> str:
    return "\n".join([line for line in s.splitlines() if line.strip() != ""])

df['content'] = df['content'].map(strip_extra_linejumps)

# Remove duplicates if any
df.drop_duplicates(inplace=True, subset="url")

print("Cleaned Dataset Shape: ")
print(df)

# Set up datetime suffix
date_suffix = datetime.datetime.strftime(datetime.datetime.now(), "%Y%m%d%H%M%S")
# TODO cambiar esto por algo independiente de la plataforma
filename = f"cleaned_{csv_file_name.split('/')[-1]}" 

print(f"Saving cleaned data to: {filename}")
with open(filename, "+w") as f:
    df.to_csv(f)