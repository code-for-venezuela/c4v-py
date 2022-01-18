"""
    Use this script to create a dataset for binary classification for a set of valid datasets 
    (One with data of type A and another with type B, for example)
    Remember that binary classification could mean assign one of two 
    labels, or one of a set of labels.
    # Parameters
        column : `str` = column to use as label, it will be checkef to be present in every dataset
        datasets : `[str]` = list of datasets to merge into a single one, they should be already well formatted
    # Return 
        A new dataset will be created with the concatenation of the provided datasets
"""
# Python imports
import sys
from datetime import datetime
from pytz import tzinfo
from pathlib import Path

# Third party imports
import pandas as pd

# Not enough arguments checking
if len(sys.argv) < 3:
    print("Not enough arguments: expected column and at the least one dataset", file=sys.stderr)
    exit(1)


# get value for column
label_column_name = sys.argv[1]

# get value for datasets 
datasets_files = sys.argv[2:]

# Validate datasets
for file in datasets_files:
    p = Path(file)
    if not p.exists():
        print(f"Invalid file: {file}", file=sys.stderr)

# read csvs
dfs = []
for file in datasets_files:
    print(f"Openning file: {file}...")
    with open(file) as f:
        new_df: pd.DataFrame = pd.read_csv(f)
    print(f"Rows in file: {len(new_df)}")

    # Check that it provides the label column
    if label_column_name not in new_df.columns:
        print(f"Dataset '{file}' does not provides the requested column: {label_column_name}", file=sys.stderr)
        exit(1)

    dfs.append(new_df)

# concat csvs
df = pd.concat(dfs)
del dfs

# Shuffle dataset
df = df.sample(frac=1).reset_index(drop=True)

# time string
date_suffix = datetime.strftime(datetime.now(), "%Y%m%d%H%M%S")
filename = f"train_dataset_{date_suffix}.csv"

print("Saving file to ", filename)
df.to_csv(filename, index=False)
