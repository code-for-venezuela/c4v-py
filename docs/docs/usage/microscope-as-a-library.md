# Microscope as a Library
You can use Microscope as a library in many ways using buth its API and its core components. 

## Using the high level Api
The main object you can use to access common operations for the Microscope library is the `microscope.Manager`
object. For example, here you can use it to crawl for urls in some known site:

```py
import c4v.microscope as ms

# creates a manager object 
manager = ms.Manager.from_default()

# crawl new urls from the internet
d = manager.crawl_new_urls_for(
    ["primicia"],               # Name of crawlers to use when crawling
    limit=10                    # Maximum ammount of urls to crawl
)

print(d)       # A (possibly empty) list of urls as string
print(len(d))  # a number <= 10
```

!!! Note
    You can find which crawler names are available for you to use using `ms.Manager.get_available_crawlers()`
---
## Examples
The following are some examples for some common use cases

### Scraping and crawling at the same time
The following code will crawl and scrape 10 urls from primicia's website 
```py
import c4v.microscope as ms

# creates a manager object 
manager = ms.Manager.from_default() 

# crawl new urls from the internet
d = manager.crawl_and_scrape_for(
    ["primicia"],                   # scrape for primicia
    limit=10                        # up to ten urls
)

print(d)            # bunch of text probably, instances of ScrapedData class
print(len(d))       # amount of scraped data instances, <= 10 as we requested limit = 10
```

### Get data for known urls
This is probably the most common operation you may want to perform, retrieving data for urls you 
want to process using this library
```py
import c4v.microscope as ms

# creates a manager object 
manager = ms.Manager.from_default() 

# Try to get data for this urls
d = manager.get_bulk_data_for(
    [
        "https://primicia.com.ve/deportes/emiliano-martinez-y-buendia-dejan-la-seleccion-argentina/",
        "https://primicia.com.ve/deportes/odubel-remolco-la-de-ganar-en-el-decimo/",
        "https://primicia.com.ve/deportes/valtteri-bottas-le-dice-adios-a-mercedes-e-ira-a-alfa-romeo/"
    ]
)

print(d) # data for the three given urls
```
### Common workflow
A common workflow for the library looks like this:
```python
import c4v.microscope as ms
from c4v.scraper.persistency_manager import SqliteManager

db = SqliteManager("my_db.sqlite")        # Replace with your custom db if necessary
manager = ms.Manager.from_default(db=db)  

# Crawl and scrape
manager.crawl_and_scrape_for(["primicia"], limit=10)

#  classify objects in database
manager.run_pending_classification_from_experiment("branch_name", "experiment_name")

# Print some results
for d in manager.get_all():
    print(f"{d.title}: {d.label.value}")

```
1. We first create our main manager using a custom database
2. Then we are crawling and scraping urls for a given site
3. And finally, we use the classifier created and located in the experiment folder `branch_name/experiment_name` to classify the results
4. And last but not least, we print the stored data along with its predicted label

---
## Using Local Storage
You can provide a database manager to store data scraped with the microscope manager locally, here we will see 
some examples using an SQLite database

### Get data for known urls
This example is the same as before, but now we will store the results directly in the database, so we can use that
data afterwards without having to scrape them.

```py
import c4v.microscope as ms
from datetime import datetime

# creates a manager object
manager = ms.Manager.from_local_sqlite_db("test_db.sqlite") # will create a file to store retrieved data

# Measure time before storing
start = datetime.now()
d = manager.get_bulk_data_for(
    [
        "https://primicia.com.ve/deportes/emiliano-martinez-y-buendia-dejan-la-seleccion-argentina/",
        "https://primicia.com.ve/deportes/odubel-remolco-la-de-ganar-en-el-decimo/",
        "https://primicia.com.ve/deportes/valtteri-bottas-le-dice-adios-a-mercedes-e-ira-a-alfa-romeo/"
    ]
)
end = datetime.now()

print("before: ", (end - start).total_seconds()) # 2.137678, May vary depending on your internet connection

# Measure time after storing
start = datetime.now()
d = manager.get_bulk_data_for(
    [
        "https://primicia.com.ve/deportes/emiliano-martinez-y-buendia-dejan-la-seleccion-argentina/",
        "https://primicia.com.ve/deportes/odubel-remolco-la-de-ganar-en-el-decimo/",
        "https://primicia.com.ve/deportes/valtteri-bottas-le-dice-adios-a-mercedes-e-ira-a-alfa-romeo/"
    ]
)
end = datetime.now()

print("after: ", (end - start).total_seconds()) #  0.000406
```

!!! Note 
    If you don't provide any db name, calling the manager constructor as `Manager.from_local_sqlite_db()` you can use 
    the library db, where data required by the CLI tool is stored by default, this way you can browse data and process
    it using code

### Retrieving data from a local db
Once you have scraped & stored data using the local SQLite manager, you may want to retrieve for further processing. 
You can do so by using the `get_all` function that returns all stored `ScrapedData` instances:

```py
import c4v.microscope as ms

# creates a manager object
manager = ms.Manager.from_local_sqlite_db("test_db.sqlite")

for d in manager.get_all():
    print(d) # prints the three instances scraped in the previous example
```

### Using your own database implementation
You might be interested in using your own persistency management strategy, you can do so by 
implementing the `BasePersistencyManager` class. For example, let's say this is our implementation:
```py
class MyDBManager(BasePersistencyManager):
    """
        Store data in the class itself 
    """
    data = set()
    def get_all(self, limit = 100, scraped = None):
        limit = 10000 if limit < 0 else limit # very high number when negative number is provided

        def goes_in(scraped_data):
            was_scraped = self.was_scraped(scraped_data.url)
            return (scraped and was_scraped) or (scraped == False and not was_scraped) or (scraped == None)
        
        return (d for d in list(MyDBManager.data)[:limit] if goes_in(d))

    def was_scraped(self, url):
        return any(d.last_scraped != None and d.url == url for d in MyDBManager.data)

    def save(self, url_data):
        MyDBManager.data = MyDBManager.data.union(url_data)

    def filter_scraped_urls(self, urls):
        scraped_urls = { d.url for d in MyDBManager.data if self.was_scraped(d.url) }
        return [url for url in urls if url not in scraped_urls]
```

This is a partial implementation, with the minimum code to save and retrieve data, let's use it as a storage backend 
for the manager class. We will use the same example as before, but with our new backend:

```py

import c4v.microscope as ms
from datetime import datetime

# creates a manager object
manager = ms.Manager(MyDBManager())

urls =  [
            "https://primicia.com.ve/deportes/emiliano-martinez-y-buendia-dejan-la-seleccion-argentina/",
            "https://primicia.com.ve/deportes/odubel-remolco-la-de-ganar-en-el-decimo/",
            "https://primicia.com.ve/deportes/valtteri-bottas-le-dice-adios-a-mercedes-e-ira-a-alfa-romeo/"
        ]

# Measure time before storing
start = datetime.now()
d = manager.get_bulk_data_for(urls)
end = datetime.now()

print("before: ", (end - start).total_seconds()) # 2.155265s, scraped from internet

# Measure time after storing
start = datetime.now()
d = manager.get_bulk_data_for(urls)
end = datetime.now()

print("after: ", (end - start).total_seconds()) #  1.7e-05s, retrieved from local storage
```
!!! Warning 
    Please not that **this is not a full implementation**, and thus, **it can't be used with the `microscope.Manager` object**
    as a database backend. If you need to do so, follow the instructions in [this](../development/creating-a-persistency-manager.md) page.

---
## Using the Low Level Api   
If you need a more fine-grained control, you can use the primary components of the microscope library, importing the following 
modules:

* `c4v.scraper` : Functions to scrape data from the internet for a given set of urls
* `c4v.scraper.crawlers` : Classes for crawling and implement a new crawler 
* `c4v.scraper.persistency_manager` : Classes for storing data locally and implement a new persistency manager
* `c4v.classifier` : Classes for classifier and its experiments:
    * Classifier Class
    * Classifier experiment class
    * Experiment Base class, if you want to create more experiments that will use the same filesystem as the rest of the experiments

More about this in the next section