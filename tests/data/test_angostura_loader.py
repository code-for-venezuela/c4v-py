from c4v.data.angostura_loader import AngosturaLoader

def test_angostura():

    test = AngosturaLoader()
    df = test.create_query(
        "SELECT * FROM `event-pipeline.angostura.sinluz_rawtweets` LIMIT 1"
    )
    df.info()
