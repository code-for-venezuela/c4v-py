"""
    This script will trim out all but the specified columns of a dataset specified by its filename, 
    and store the result in another file. 
    If you want to create a regular dataset with a specified label field
    # Arguments
        dataset : `str` = filename of the dataset whose rows will be removed 
        -l label_name : `str` = You can optionally provide a field name after the flag -l to mark it as a label field, so that
                                it can be used as label in a classification pipeline. If present, it should be right before
                                the column list
        columns : `[str]` = list of names of columns that will be kept, they should be 
                            valid columns and an error will be raised if not
    # Return
        A new file will be written in the current directory with the new dataset.
"""

# Python imports
import sys
from pathlib import Path

# Third party imports
import pandas as pd

# -- < Input & sanity check > ---------------------------

# Utility function that will log an error and end the program  execution with a non-zero exit code
def raise_error(msg : str):
    print(f"[ERROR] {msg}", file=sys.stderr)
    exit(1)


# should provide at the least name of this file, path to dataset, and at the least one column to remove
if len(sys.argv) < 3: 
    raise_error("Expected at the least 2 arguments, path to dataset as csv and a non-empty list of columns to remove")

# Try to parse label flag
if sys.argv[2] == '-l' and len(sys.argv) < 5:
    raise_error("Expected at the least 3 arguments, path to dataset as csv, label flag value and a non-empty list of columns to remove")
elif sys.argv[2] == "-l":
    label_column = sys.argv[3]
    columns_start_index = 4
else:
    label_column = None
    columns_start_index = 2

print("label column is:", label_column)

# Parse file and columns
file = sys.argv[1]
columns = sys.argv[columns_start_index:]

# Check that the given file is a valid one
path_to_file = Path(file)

if not path_to_file.exists():
    raise_error("The given file does not exists")
elif not file.endswith(".csv"):
    raise_error("This doesn't seems to be a valid csv file")

# Load dataset
dataframe = pd.read_csv(file)

# Get dataset columns
actual_columns = list(dataframe.columns)

# Check consistent columns 
if any(x not in actual_columns for x in columns) or ( False if not label_column else (label_column not in actual_columns)):
    raise_error("Inconsistent columns and csv. Provided columns not in csv format")

# -- < filter out undesired columns > ---------------------
print("Deleting columns...")
columns_to_drop = [c for c in actual_columns if c not in columns]
dataframe.drop(columns_to_drop, axis='columns', inplace=True)

# Rename column to label if requested so
if label_column:
    print(f"Renaming column {label_column} to 'label'...")
    dataframe.rename({label_column : 'label'}, axis='columns', inplace=True)

# -- < Save to file > -------------------------------------
filename = path_to_file.stem + "_trimmed_cols.csv"
print(f"Saving new csv to: {filename}...")

dataframe.to_csv(filename, index=False)
