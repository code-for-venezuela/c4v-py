"""
    This file will take a list of csv files and merge them in a single one 
    ready for training
"""
import pandas as pd
import sys

files = sys.argv[1:]

# Check if no files to work with
if not files:
    print("No files to merge")
    exit(0)

# Open every file and load dataframes
dfs = []
for file in files:
    print(f"Openning file: {file}...")
    with open(file) as f:
        new_df: pd.DataFrame = pd.read_csv(f)
    print(f"Rows in file: {len(new_df)}")
    dfs.append(new_df)

# Remove irrelevant columns
columns = ["url", "title", "content", "author", "categories", "date", "label", "source"]

# Concate dataframes
df = pd.concat(dfs)
del dfs

# Remove irrelevant columns
columns_to_remove = [col for col in df.columns if col not in columns]
print(f"Removing columns: {columns_to_remove}")

print("Resulting dataframe: ")
print(df)


# Saving dataframe
filename = "aggregated_dataframe.csv"
print(f"Saving csv to: {filename}")
df.to_csv(filename, index=False)
