from c4v.data.angostura_loader import AngosturaLoader
import tensorflow as tf
import pandas as pd
import math
import os


class TweetLoader:
    def __init__(self):
        """
        Abstraction of AngosturaLoader to programmatically query Angostura's tweets tables.
        """
        self.angostura_loader = AngosturaLoader()

    def load_sinluz_tweets(
        self, batch_size, batch_number, print_table_info=False, return_tf_dataset=True
    ):
        """
        Queries `angostura.meditweet_sin_luz`.
        Receives batch_size and number of batches to query.
        Yields a generator that returns the queried data.
        ----------------------------------------------------

        Params: 

            batch_size: Int. Default = None.
                Number of tweets to query per batch.
            
            batch_number: Int. Default = None.
                Number of batches to query.

            print_table_info: Bool. Default = False.
                Prints number of records in `angostura.meditweet_sin_luz`.

            return_tf_dataset: Bool. Default = True.
                If True, generator returns a tf.Dataset only containing the column `full_text`.
                If False, generator returns a dataframe with the complete number of columns:
                    id, created_at, full_text, metadata,  user, geo, place

        Example:

            test = TweetLoader()
            test_query = test.load_sinluz_tweets(batch_size=100, batch_number=1)
            
            # Print values
            for x in next(test_query):
                    print(x)

            test_query = test.load_sinluz_tweets(batch_size=100, batch_number=2)
            
            # Loop over queried batches
            for text_batch in test_query:
                print(text_batch)
        """

        # I added this due to a bug -  Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX AVX2
        # Fix Reference: https://stackoverflow.com/questions/47068709/your-cpu-supports-instructions-that-this-tensorflow-binary-was-not-compiled-to-u
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

        if print_table_info:
            total_tweets = self.angostura_loader.create_query(
                """
                SELECT 
                    count(*) AS count_total
                FROM 
                    `angostura.meditweet_sin_luz` 
                """
            )
            total_records = int(total_tweets.count_total)

            print(
                f"The table `angostura.meditweet_sin_luz` has a total of {total_records} rows."
            )

        for batch in range(batch_number):

            print(f"Loading batch number {batch}")

            query = f"""
                SELECT 
                    *
                FROM 
                    `angostura.meditweet_sin_luz`
                ORDER BY 
                    created_at ASC
                LIMIT {batch_size}
                """

            batch_df = self.angostura_loader.create_query(query)
            target = batch_df.pop("full_text").values

            text_tfdataset = tf.data.Dataset.from_tensor_slices(target)

            if return_tf_dataset:
                yield text_tfdataset
            else:
                yield batch_df
