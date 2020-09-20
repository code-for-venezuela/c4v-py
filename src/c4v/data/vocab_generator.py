# import sys
# sys.path.append(r'../')
import os
import pandas as pd
import tensorflow_datasets as tfds


from c4v.data.data_cleaner import DataCleaner


# set path for resources
path_to_corpus = "../../../data/raw/tweets/tagging-set-original_for_jupyter_tagging.csv"
vocab_filename = "vocab"
path_to_vocab = os.path.join("../../../data/processed", vocab_filename)

# read from the csv
corpus = pd.read_csv(path_to_corpus)

# clean the text
cleaned_corpus = DataCleaner.data_prep_4_vocab(corpus["full_text"])

# show the tweets after they have been "cleaned"
for tweet, clean_tweet in zip(corpus["full_text"].to_list(), cleaned_corpus.to_list()):
    print("---<START>")
    print("\t", tweet)
    print("\t", clean_tweet)
    print("<END>---")


# Build and save the vocab
# encoder = tfds.features.text.SubwordTextEncoder.build_from_corpus(
#     cleaned_corpus.to_list(), target_vocab_size=10000
# )

# encoder.save_to_file(path_to_vocab)
# print("size: ", encoder.vocab_size)


