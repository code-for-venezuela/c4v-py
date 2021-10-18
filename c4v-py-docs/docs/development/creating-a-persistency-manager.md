# Creating a Persistency Manager
The Persistency manager is an object required by the high level manager object `microscope.Manager` in order to access the data in which it will 
work. Such data can be obtained in many ways and stored in even more ways. So, in order to support multiple storage alternatives, we provide the `BasePersistencyManager` class, an interface that should be implemented by any Persistency Manager so it can be used to retrieve data, and passed to the `microscope.Manager` to tell it how to access the desired data. In this page we will describe how to create your own manager.  

In the following example we will implement a dict-based persistency manager.

# Implementing a new Persistency Manager class
First thing first, we have to import the base class that provides the interface to be implemented. 
```python
from c4v.scraper.persistency_manager import BasePersistencyManager

class DictManager(BasePersistencyManager):
    """
        Persistency Manager class that stores data in a dict object
    """
```
!!! Warning
    Remember always document the class to have a bit more context about what it does

The only required methods to be implemented are the ones marked with `NotImplementedError`, which (for now) are actually all of them.

## Constructor:
First we do all the required initialization code in the constructor, don't forget to call the parent constructor:
```python
from typing import Dict
from c4v.scraper.persistency_manager import BasePersistencyManager
from c4v.scraper import ScrapedData

class DictManager(BasePersistencyManager):
    """
        Persistency Manager class that stores data in a dict object
    """

    def __init__(self):
        super().__init__() # Don't forget to call the parent class constructor
        _stored_data : Dict[str, ScrapedData] = {}
```

## Get All
This function is your main interface function to access stored data. Read carefully the description in the `BasePersistencyManager` class to know what is expected for this function to do.

>   Return an iterator over the set of stored instances   
>   **Parameters**:   
>       + **limit** : `int` = Max amount of elements to retrieve. If negative, no limit is assumed.
>       + **scraped** : `bool` = True if retrieved data should be scraped, false if it shouldn't, None if not relevant   
>       + **order_by** : `str` = (optional) names of the fields to use for sorting, first char should be order, - for descending, + for ascending, following chars in each string should be a valid name of a field in the ScrapedData dataclass. If no provided, no order is ensured   
>   **Return**:   
>       Iterator of stored ScrapedData instances

We start by creating the function and adding some sanity check. Remember that this is possibly going to be useful in the future
```python
from typing import Dict, Iterator, List
from c4v.scraper.persistency_manager import BasePersistencyManager
from c4v.scraper import ScrapedData
from dataclasses import fields

def get_all(self, limit: int, scraped: bool, order_by: List[str] = None) -> Iterator[ScrapedData]:
        # Remember to do some sanity check
        valid_fields = {f.name for f in fields(ScrapedData)}
        for order in order_by:
            assert order and order[0] in ["-", "+"] and order[1:] in  valid_fields, "not valid order provided: " + order
```

Now we retrieve and filter the stored data:
```python
def get_all(self, limit: int, scraped: bool, order_by: List[str] = None) -> Iterator[ScrapedData]:
    # Sanity check...
    # Get actual data, filtering by scraped or not
    # This lambda is checking if the data instance should be included in the query.
    # We assume that an instance is not scraped when its last_scraped field is not provided
    # (AKA it wasn't scraped in any moment)
    goes_in = lambda x: scraped == None or \
                        (x.last_scraped and scraped) or \
                        (not x.last_scraped and not scraped) 
    data = [d for d in self._stored_data.values() if goes_in(d)]
```