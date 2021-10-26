from c4v.scraper.scraped_data_classes.scraped_data  import ScrapedData, ScrapedDataEncoder
from c4v.config                                     import settings
import json
import datetime
import pytz
from dataclasses import asdict

DATE_FORMAT = settings.date_format

def test_scraped_data_encoding():
    date = datetime.datetime(2020, 3,30,0,0,0, tzinfo=pytz.UTC)

    sample_data1 = ScrapedData("www.c4v.com")
    sample_data2 = ScrapedData("www.arepa.com", date, "Nuevo tipo de arepa", "Ahora con carne mechada", "Michiberto Rodriguez", ["areas", "mechada"], "2020-02-02")

    expected_output1 = asdict(sample_data1)
    expected_output2 = asdict(sample_data2)
    expected_output2['last_scraped'] = datetime.datetime.strftime(expected_output2['last_scraped'], DATE_FORMAT)


    assert json.loads(json.dumps(sample_data1, cls=ScrapedDataEncoder)) == expected_output1
    assert json.loads(json.dumps(sample_data2, cls=ScrapedDataEncoder)) == expected_output2