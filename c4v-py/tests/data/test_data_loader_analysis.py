# Import class

import sys
sys.path.append('../helper_functions/')
import data_loader
import numpy as np
import pandas as pd

#Load Data into class and parse dataframe

test = data_loader.load_data('../brat-v1.3_Crunchy_Frog/data/first-iter/sampled_58_30')


# Print parsed dataframe

print(test.parsed_df.head())


# Vectorize text and train test split
    # This method initialized following attributes:
        # X_train, X_test, y_train, y_test, X, y
test.preprocess()

# Print X_train shape 
print(test.X_train.shape)
