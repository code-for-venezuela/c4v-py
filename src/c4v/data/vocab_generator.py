# import sys
# sys.path.append(r'../')
import os
import pandas as pd
import tensorflow_datasets as tfds


def cleaner(df: pd.DataFrame) -> pd.DataFrame:
    """
    This method is an improved copy of the oe used in data_sampler.py
    Todo: improve the OOD of these classes
    """
    df = df.str.lower()

    # Convert common spanish accents

    df = df.str.replace("ú", "u")
    df = df.str.replace("ù", "u")
    df = df.str.replace("ü", "u")
    df = df.str.replace("ó", "o")
    df = df.str.replace("ò", "o")
    df = df.str.replace("í", "i")
    df = df.str.replace("ì", "i")
    df = df.str.replace("é", "e")
    df = df.str.replace("è", "e")
    df = df.str.replace("á", "a")
    df = df.str.replace("à", "a")
    df = df.str.replace("ñ", "gn")

    # Remove links
    df = df.str.replace("http.+", " ")

    # Remove Punctuation
    df = df.str.replace("[\\.\\-:,\\?]", " ")

    # Remove white spaces
    df = df.str.replace(r"[\s]+", " ")

    # I need to remove all spaces before and after each string
    df = df.str.strip()

    return df


# set path for resources
path_to_corpus = "../../../data/raw/tweets/tagging-set-original_for_jupyter_tagging.csv"
vocab_filename = "vocab"
path_to_vocab = os.path.join("../../../data/processed", vocab_filename)

# read from the csv
corpus = pd.read_csv(path_to_corpus)

# clean the text
cleaned_corpus = cleaner(corpus["full_text"])

# Build and save the vocab
encoder = tfds.features.text.SubwordTextEncoder.build_from_corpus(
    cleaned_corpus.to_list(), target_vocab_size=10000
)

encoder.save_to_file(path_to_vocab)
# print("size: ", encoder.vocab_size)

# show the tweets after they have been "cleaned"
for tweet in cleaned_corpus.to_list():
    print("---<START>", tweet, "<END>---")
