from c4v.data.tweet_loader import TweetLoader


def test_tweet_loader():

    test = TweetLoader()
    test_query = test.load_sinluz_tweets(batch_size=20, batch_number=2)

    # Loop over queried batches
    for text_batch in test_query:
        print(text_batch)
