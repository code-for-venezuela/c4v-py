# Import interface class
from typing import Dict, Iterator, List
from c4v.scraper.persistency_manager import BasePersistencyManager
from c4v.scraper import ScrapedData
from dataclasses import fields

class DictManager(BasePersistencyManager):
    """
        Persistency Manager class that stores data in a dict object
    """

    def __init__(self):
        super().__init__() # Don't forget to call the parent class constructor
        self._stored_data : Dict[str, ScrapedData] = {}

    def get_all(self, limit: int = -1, scraped: bool = None, order_by: List[str] = None) -> Iterator[ScrapedData]:        

        # Remember to do some sanity check
        valid_fields = {f.name for f in fields(ScrapedData)}
        if order_by:
            for order in order_by:
                assert order and order[0] in ["-", "+"] and order[1:] in  valid_fields, "not valid order provided: " + order

        # Get actual data, filtering by scraped or not
        # This lambda is checking if the data instance should be included in the query.
        # We assume that an instance is not scraped when its last_scraped field is not provided
        # (AKA it wasn't scraped in any moment)
        goes_in = lambda x: scraped == None or \
                            (x.last_scraped and scraped) or \
                            (not x.last_scraped and not scraped) 
        data = [d for d in self._stored_data.values() if goes_in(d)]

        # Now sort it as requested
        order_by = order_by or []
        for field in order_by:
            asc = field[0] == "+" 
            data.sort(key=lambda d: d.__getattribute__(field[1:]), reverse=not asc)
        print(data)

    def save(self, url_data: List[ScrapedData]):
        for d in url_data:
            self._stored_data[d.url] = d

### TEST ZONE, DELETE LATER
import datetime
datas = [
    ScrapedData("www.michimon.com"),
    ScrapedData("www.hl3confirmed.com", last_scraped=datetime.datetime.now(), title="I Want To Believe", content="Not yet but soon", author="ubuntuOS")
]
dm = DictManager()
dm.save(datas)
dm.get_all( order_by=["-url"])

