from angostura_loader import AngosturaLoader
import pandas as pd
import math


class TweetLoader:
    def __init__(self):

        self.angostura_loader = AngosturaLoader()

    def load_sinluz_tweets(self, batch_size, batch_number, print_table_info=False):

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

            print("Loading batch number {batch}")

            query = f"""
                SELECT 
                    *
                FROM 
                    `angostura.meditweet_sin_luz`
                ORDER BY 
                    created_at ASC
                LIMIT {batch_size}
                """

            yield self.angostura_loader.create_query(query)


if __name__ == "__main__":

    test = TweetLoader()
    test_query = test.load_sinluz_tweets(batch_size=100, batch_number=2)
    print(next(test.load_sinluz_tweets(batch_size=100, batch_number=2)))
    print(
        next(
            test.load_sinluz_tweets(
                batch_size=100, batch_number=2, print_table_info=True
            )
        )
    )
