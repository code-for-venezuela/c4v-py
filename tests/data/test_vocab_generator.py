import pandas as pd
import tensorflow as tf
import os

from c4v.data.tweet_loader import TweetLoader
from c4v.data.vocab_generator import Vocabulary

FULL_TEXT_LABEL = "full_text"
PATH_TO_SAMPLE_CORPUS = "../../data/raw/tweets/tagging-set-original_for_jupyter_tagging.csv"


def test_vocab_generator():
    # sin_luz_raw_tweets = _read_tweets_from_source("tw")
    data = _read_tweets_from_source()
    path = os.path.join('spanish_vocabulary')
    voc = Vocabulary(data[FULL_TEXT_LABEL], 500, path)
    encoder = voc.generate_and_save()

    tweet = "La luz se va todos los dias y asi no se puede vivir"
    ids = encoder.encode(tweet)
    text = encoder.decode([1, 2, 3, 4])
    print(f'example: {ids}\n result: {text}')


def _read_tweets_from_source(source_type: str = "csv"):
    corpus = pd.DataFrame()
    if source_type == "csv":
        # read from the csv
        corpus = pd.read_csv(PATH_TO_SAMPLE_CORPUS)
    elif source_type == "tw":
        tw = TweetLoader()
        generator = tw.load_sinluz_tweets(
            batch_size=5000, batch_number=4, return_tf_dataset=False
        )
        for tweet in generator:
            if isinstance(tweet, pd.DataFrame) or isinstance(tweet, pd.Series):
                corpus.append(tweet[FULL_TEXT_LABEL])
                print("\t", corpus.size)
            elif isinstance(tweet, tf.data.Dataset):
                print("\tDo something here")
            else:
                raise TypeError(
                    "each item of the generator must be a dataframe or tfdset type"
                )

    return corpus
#
#
# def show_cleaned_data(raw_data: pd.DataFrame, cleaned_data: pd.DataFrame):


#     # show the tweets after they have been "cleaned"
#     for tweet, clean_tweet in zip(
#         raw_data[FULL_TEXT_LABEL].to_list(), cleaned_data.to_list()
#     ):
#         print("---<START>")
#         print("\t", tweet)
#         print("\t", clean_tweet)
#         print("<END>---")
#
#


if __name__ == "__main__":
    corpus = pd.DataFrame([])
    tw = TweetLoader()
    generator = tw.load_sinluz_tweets(
        batch_size=5000, batch_number=4, return_tf_dataset=False
    )
    for tweet in generator:
        if isinstance(tweet, pd.DataFrame) or isinstance(tweet, pd.Series):
            corpus.append(tweet[FULL_TEXT_LABEL])
            print("\t", corpus.size)
        elif isinstance(tweet, tf.data.Dataset):
            print("\tDo something here")
        else:
            raise TypeError(
                "each item of the generator must be a dataframe or tfdset type"
            )