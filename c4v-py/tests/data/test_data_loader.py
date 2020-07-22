from c4v.data.data_loader import BratDataLoader

def test_all():
    loader = BratDataLoader(['./brat-v1.3_Crunchy_Frog/data/first-iter/balanced_dataset_brat'])
    df = loader.get_parsed_df()
    print(df.info())
