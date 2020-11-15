# import sys
# sys.path.append(r'../')
import os
import logging

import tensorflow_datasets as tfds

import pandas as pd
from c4v.data.data_cleaner import DataCleaner

# set path for resources
# PATH_TO_SAMPLE_CORPUS = "../../../data/raw/tweets/tagging-set-original_for_jupyter_tagging.csv"
# PROCESSED_DATA_FOLDER = "../../../data/processed"


class Vocabulary:
    """
    Generates a vocabulary using byte pair encoding
    """
    def __init__(self, corpus, vocab_size=10000, vocab_filepath=None):
        """
               Uses byte pair encoding provided by tensorflow to generate a vocabulary.
               Receives a corpus, vocab_size and vocab_filepath.
               Yields an encoder ready to be used based on the vocabulary generated.
               ----------------------------------------------------

               Params:

                   corpus: pd.DataFrame Required.
                       The corpus that will be used to generate the vocabulary.

                   vocab_size: Int. Default = 10000.
                       Sets the number of elements to be generated as part of the vocabulary.

                   vocab_filepath: Str. Default = None.
                       Filepath used to store the vocabulary.  If None provided, the vocabulary
                       will be stored on the path from which the class was invoked under the name
                       vocab_[vocab_size].

               Example:
                   data = pd.DataFrame()
                   voc = Vocabulary(data, 500, path)
                   encoder = voc.generate_and_save()

                   tweet = "Mi mama me cocino mucha comida hoy"
                   ids = encoder.encode(tweet)
                   text = encoder.decode([1, 2, 3, 4])
                   print(f'example: {ids}\n result: {text}')
                   """
        self.logger = logging.getLogger(__name__)

        self.corpus = corpus
        self.size = vocab_size
        self.filepath = self._vocab_path(vocab_filepath)

        self._clean_corpus = None

    def get_clean_corpus(self):
        return self._clean_corpus

    def _vocab_path(self, filepath):
        if filepath is None:
            return os.path.join("./", "vocab_" + str(self.size))
        else:
            return os.path.join(filepath)

    def _generate_vocab(self):
        # clean the text
        self.clean_corpus = DataCleaner.data_prep_4_vocab(self.corpus)

        # Build and save the vocab
        encoder = tfds.deprecated.text.SubwordTextEncoder.build_from_corpus(
            self.clean_corpus.to_list(), target_vocab_size=self.size
        )
        return encoder

    def _save_vocab(self, encoder):
        # Save the vocabulary
        encoder.save_to_file(self.filepath)
        self.logger.info("saved: ", self.filepath)
        self.logger.info("size: ", self.size)

    def generate_and_save(self):
        """
        Generated the vocabulary, saves it on the designated location and returns the encoder.
        """
        new_encoder = self._generate_vocab()
        self._save_vocab(new_encoder)
        return new_encoder


# if __name__ == '__main__':
#
#     data = pd.read_csv(PATH_TO_SAMPLE_CORPUS)
#     path = os.path.join(PROCESSED_DATA_FOLDER, 'spanish_vocabulary')
#     voc = Vocabulary(data, 500, path)
#     en = voc.generate_and_save()
