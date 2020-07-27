from c4v.data.data_loader import BratDataLoader

#Load Data into class and parse dataframe

test = BratDataLoader(['data/processed/brat/sampled_58_30'])


# Print parsed dataframe
print(test.parsed_df.head())


# Vectorize text and train test split
    # This method initialized following attributes:
        # X_train, X_test, y_train, y_test, X, y
test.preprocess()

# Print X_train shape 
print(test.X_train.shape)
