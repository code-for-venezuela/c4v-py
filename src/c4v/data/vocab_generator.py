# import sys
# sys.path.append(r'../')
import os, sys
import pandas as pd
import tensorflow_datasets as tfds

from c4v.data.data_cleaner import DataCleaner
from c4v.data.angostura_loader import AngosturaLoader

# set path for resources
PATH_TO_COURPUS = "../../../data/raw/tweets/tagging-set-original_for_jupyter_tagging.csv"
FULL_TEXT_LABEL = "full_text"
VOCAB_FILENAME = "vocab"
PATH_TO_VOCAB = os.path.join("../../../data/processed", VOCAB_FILENAME)



def read_from_angostura():
    test = AngosturaLoader()
    df = test.create_query(
        "SELECT * FROM `event-pipeline.angostura.sinluz_rawtweets` LIMIT 1"
    )
    df.info()
    return df


def read_tweets_from_source(source_type: str = 'csv'):
    if source_type == 'csv':
        # read from the csv
        corpus = pd.read_csv(PATH_TO_COURPUS)
    elif source_type == 'bq':
        corpus = read_from_angostura()
    else:
        raise Exception('not a valid type of source. csv or bq for big query sources')

    return corpus


def show_cleaned_data(raw_data, cleaned_data):
    # show the tweets after they have been "cleaned"
    for tweet, clean_tweet in zip(raw_data[FULL_TEXT_LABEL].to_list(), cleaned_data.to_list()):
        print("---<START>")
        print("\t", tweet)
        print("\t", clean_tweet)
        print("<END>---")


def generate_vocab():
    # read data
    try:
        corpus = read_tweets_from_source(source_type='bq')

        # clean the text
        cleaned_corpus = DataCleaner.data_prep_4_vocab(corpus[FULL_TEXT_LABEL])

        # This is a debugging line
        show_cleaned_data(corpus, cleaned_corpus)

        # Build and save the vocab
        encoder = tfds.features.text.SubwordTextEncoder.build_from_corpus(
            cleaned_corpus.to_list(), target_vocab_size=10000
        )

        # Save the vocabulary
        encoder.save_to_file(PATH_TO_VOCAB)
        print("size: ", encoder.vocab_size)

    except FileNotFoundError:
        type_, value, traceback = sys.exc_info()
        print(f'Error: {value}')

    except:
        type_, value, traceback = sys.exc_info()
        print(f'Error: {value}')


if __name__ == '__main__':
    generate_vocab()
