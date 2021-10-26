from c4v.data.data_loader import BratDataLoader

def test_all():
    loader = BratDataLoader(['data/processed/brat/balanced_dataset_brat'])
    df = loader.get_parsed_df()
    print(df.info())
