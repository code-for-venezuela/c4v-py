"""
    Use this script to create a dataset for binary classification.
    Remember that binary classification could mean assign one of two 
    labels, or one of a set of labels.
    # Parameters
        column : `str` = column to use as label
        label_set : `str` = labelset name to use for classification. Options are:
            - SERVICE    : for 'servicio' column
            - ISSUE_TYPE : for 'tipo 
"""




old_df = df.copy()
url_to_labels_df = old_df.groupby("url")

# Remove duplicates
df.drop_duplicates(inplace=True, ignore_index=True, subset=["url"])

# Try to edit label field
for (url, sub_df) in url_to_labels_df:
    labels = list(set(sub_df.label))
    df.loc[df.url == url, "label"] = [labels]
