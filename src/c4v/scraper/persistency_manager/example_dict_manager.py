# Import interface class
from typing import Dict, Iterator, List
from c4v.scraper.persistency_manager import BasePersistencyManager
from c4v.scraper import ScrapedData
from dataclasses import fields
import sys

class DictManager(BasePersistencyManager):
    """
        Persistency Manager class that stores data in a dict object
    """

    def __init__(self):
        super().__init__() # Don't forget to call the parent class constructor
        # We will store ScrapedData instances in a dict, using its 
        # url as a key
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
        
        # Set up limit 
        # All elements by default
        limit = limit if limit > 0 else len(data)

        for i in range(limit):
            yield data[i]

    def filter_known_urls(self, urls: List[str]) -> List[str]:
        # Just return the ones that are not stored in our dict
        return [url for url in urls if not self._stored_data.get(url)]

    def filter_scraped_urls(self, urls: List[str]) -> List[str]:
        # Just return the ones that are either not stored, or stored but not yet scraped
        return [
            url for url in urls 
                if not self._stored_data.get(url) or\
                not self._stored_data[url].last_scraped
            ]

    def was_scraped(self, url: str) -> bool:
        # Return true if it's stored in DB and it was scraped at some point
        return bool(self._stored_data.get(url) and self._stored_data[url].last_scraped)

    def save(self, url_data: List[ScrapedData]):
        # Add data instance to dict, use its url as key
        for d in url_data:
            self._stored_data[d.url] = d

    def delete(self, urls: List[str]):
        # Just remove these urls from storage dict
        for url in urls:
            if self._stored_data.get(url):
                del self._stored_data[url]

### TEST ZONE, DELETE LATER
import datetime
datas = [
    ScrapedData("www.michimon.com"),
    ScrapedData("www.hl3confirmed.com", last_scraped=datetime.datetime.now(), title="I Want To Believe", content="Not yet but soon", author="ubuntuOS")
]
dm = DictManager()
dm.save(datas)
print(list(dm.get_all()))
dm.delete(["www.hl3confirmed.com"])
print(list(dm.get_all()))