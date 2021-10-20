# Creating a Persistency Manager
The Persistency manager is an object required by the high level manager object `microscope.Manager` in order to access the data in which it will 
work. Such data can be obtained in many ways and stored in even more ways. So, in order to support multiple storage alternatives, we provide the `BasePersistencyManager` class, an interface that should be implemented by any Persistency Manager so it can be used to retrieve data, and pass it to the `microscope.Manager` to tell it how to access the desired data. In this page we will describe how to create your own manager.  

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
First we add the required initialization code in the constructor, don't forget to call the parent constructor:
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
        # We will store ScrapedData instances in a dict, using its 
        # url as a key
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

class DictManager(BasePersistencyManager):
    """
        Persistency Manager class that stores data in a dict object
    """
    # __init__...

    def get_all(self, limit: int, scraped: bool, order_by: List[str] = None) -> Iterator[ScrapedData]:
            # Remember to do some sanity check
            valid_fields = {f.name for f in fields(ScrapedData)}
            for order in order_by:
                assert order and order[0] in ["-", "+"] and order[1:] in  valid_fields, "not valid order provided: " + order
```

Now we retrieve and filter the stored data:
```python
from typing import Dict, Iterator, List
from c4v.scraper.persistency_manager import BasePersistencyManager
from c4v.scraper import ScrapedData
from dataclasses import fields

class DictManager(BasePersistencyManager):
    """
        Persistency Manager class that stores data in a dict object
    """
    # __init__...
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
        # Now sort it as requested
        order_by = order_by or []
        for field in order_by:
            asc = field[0] == "+" 
            data.sort(key=lambda d: d.__getattribute__(field[1:]), reverse=not asc)
```

And finally, we take only the requested amount of elements
```python
from typing import Dict, Iterator, List
from c4v.scraper.persistency_manager import BasePersistencyManager
from c4v.scraper import ScrapedData
from dataclasses import fields

class DictManager(BasePersistencyManager):
    """
        Persistency Manager class that stores data in a dict object
    """
    # __init__...
    def get_all(self, limit: int, scraped: bool, order_by: List[str] = None) -> Iterator[ScrapedData]:
        # Sanity check...
        # Data collecting and ordering...

        # Set up limit 
        # All elements by default
        limit = limit if limit > 0 else len(data)

        for i in range(limit):
            yield data[i]

```
## Filter known urls:
This function is required in order to check which urls are already in database, scraped or not.
This is important to efficiently check if a newly discovered url should be saved in the database.  

From the function description:
> **Filter out** urls that are **already known** to the database, leaving only 
> **the ones that are new**

```python
from typing import Dict, Iterator, List
from c4v.scraper.persistency_manager import BasePersistencyManager
from c4v.scraper import ScrapedData
from dataclasses import fields

class DictManager(BasePersistencyManager):
    """
        Persistency Manager class that stores data in a dict object
    """
    # __init__...
    # get_all...
    def filter_known_urls(self, urls: List[str]) -> List[str]:
        # Just return the ones that are not stored in our dict
        return [url for url in urls if not self._stored_data.get(url)]
```
## Filter scraped urls
This function is useful when you want to know if a given set of urls is actually already 
scraped.

From the function description:
> **Filter out** urls whose data is **already known**, leaving only the ones to be **scraped**   
> **for first time**   
> **Parameters**:    
> + **urls** : `[str]` = List of urls to filter    
> **Return**:   
>     A list of urls such that **none of them has been scraped**    

```python
from typing import Dict, Iterator, List
from c4v.scraper.persistency_manager import BasePersistencyManager
from c4v.scraper import ScrapedData
from dataclasses import fields

class DictManager(BasePersistencyManager):
    """
        Persistency Manager class that stores data in a dict object
    """
    # __init__...
    # get_all...
    # filter_known_urls...

   def filter_scraped_urls(self, urls: List[str]) -> List[str]:
        # Just return the ones that are either not stored, or stored but not yet scraped
        return [
            url for url in urls 
                if not self._stored_data.get(url) or\
                not self._stored_data[url].last_scraped
            ] 
```

## was scraped
This function is useful to tell if a given url corresponds to an actual article that it's
scraped and stored in the underlying database (a dict in our case).

From the function description:

>   Tells if a given **url** is **already scraped** (it's related data is already know)   
>   **Parameters**:   
>       + **url** : `str` = url to check if it was **already scraped**   
>   **Return**:   
>       If the given url's related data is already known   

```python
from typing import Dict, Iterator, List
from c4v.scraper.persistency_manager import BasePersistencyManager
from c4v.scraper import ScrapedData
from dataclasses import fields

class DictManager(BasePersistencyManager):
    """
        Persistency Manager class that stores data in a dict object
    """
    # __init__...
    # get_all...
    # filter_known_urls...
    # filter_scraped_urls... 

    def was_scraped(self, url: str) -> bool:
        # Return true if it's stored in DB and it was scraped at some point
        return bool(self._stored_data.get(url) and self._stored_data[url].last_scraped)

```

## save
This function is used to add new data to the underlying storage. It will override any 
existing data.

From the function description:


> **Save provided data** to underlying storage. 
> If some some urls are **already in local storage**, **override them** with provided new data.
> If not, **just add them**.   
> **Parameters**:   
>     + **data** : `[ScrapedData]` = data to be saved, will **override existent data**    
        
```python
from typing import Dict, Iterator, List
from c4v.scraper.persistency_manager import BasePersistencyManager
from c4v.scraper import ScrapedData
from dataclasses import fields

class DictManager(BasePersistencyManager):
    """
        Persistency Manager class that stores data in a dict object
    """
    # __init__...
    # get_all...
    # filter_known_urls...
    # filter_scraped_urls... 
    # was_scraped...
    def save(self, url_data: List[ScrapedData]):
        # Add data instance to dict, use its url as key
        for d in url_data:
            self._stored_data[d.url] = d
```

## Delete
This function will remove listed data by url. If some url is not 
actually stored, it will ignore it. 

From the function description:


>    **Delete** provided urls from persistent storage. If does not exists, ignore it.   
>    **Parameters**:   
>        **urls** : `[str]` = Urls to be deleted   

```python
from typing import Dict, Iterator, List
from c4v.scraper.persistency_manager import BasePersistencyManager
from c4v.scraper import ScrapedData
from dataclasses import fields

class DictManager(BasePersistencyManager):
    """
        Persistency Manager class that stores data in a dict object
    """
    # __init__...
    # get_all...
    # filter_known_urls...
    # filter_scraped_urls... 
    # was_scraped...
    # save...

    def delete(self, urls: List[str]):
        # Just remove these urls from storage dict
        for url in urls:
            if self._stored_data.get(url):
                del self._stored_data[url]
```
# Testing 
Last but not least, if you are working on the library source code, don't forget to test your brand new `PersistencyManager` object. There's plenty of useful utility functions to easily add testing for a new manager object, but don't forget to add more tests for your specific implementation if it requires it.

## Creating a testing file
Go to the `tests/scraper/persistency_manager/` folder from the project root and create a new file for your new persistency manager
class, let's say `test_example_manager.py`   
!!! Warning
    Remember to start every test file with the **"test_"** prefix to every test file.

## 