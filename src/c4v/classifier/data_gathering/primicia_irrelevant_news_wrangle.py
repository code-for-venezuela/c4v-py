"""
    Use this script to process output data from the primicia_irrelevant_news_scraping.py script
    The file, in order to match the one used for trainning and provided by the client, should provide the following collumns
    * Label
    * title
    * text
    * date
    * author
    * url

    Expected arguments:
        csv_file_name : str = name of csv file with corresponding dataframe to use as basis
"""
import pandas as pd
import importlib.resources as resources
import sys

from pandas.core.frame import DataFrame

if len(sys.argv) < 2:
    print("Missing csv filename argument", file=sys.stderr)
    exit(1)

csv_file_name = sys.argv[1] # name of csv file
with open(csv_file_name) as file:
    df : DataFrame = pd.read_csv(file)

# Add label column as irrelevant for every row
df['label'] = "IRRELEVANTE"

# 

print(df)