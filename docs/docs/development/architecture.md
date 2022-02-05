# Architecture & components

The Microscope library is compound by components that can be summarized as:    

* **Scraper**: Will scrape data from **known** urls for specific websites, **not every website might be scrapable**, 
returning `ScrapedData` instances, this is the scheme for the data expected from a page
* **Crawler**: Will crawl new urls from specific sources, this data should be fed to the scraper at some point
* **Persistency Manager**: Will store data scraped by the scraper in some persistent storage, an SQLite-based
manager is provided by default
* **Classifier**: Classifies a `ScrapedData` instance telling if it is a public service problem or not.
* **Experiment**: This class controls an experiment run, it's useful to manage logging and results for experiments. Also, 
it makes possible for every experiment to be ran in more or less the same way, making it easier to use for new comers.
* **ExperimentFSManager**: Simple class controlling how to experiment's filesystems are stored, enabling an unified filesystem 
for every experiment. You can implement a new object with the same interface if you want to provide an alternative method 
experiment's storage

<p align="center">
  <img src= "../../img/microscope_architecture.png">
</p>

!!! Warning
    The **classifier** should be more specific in the future, it should be able not only to differentiate between news talking 
    about public services or not, but also the kind of problem itself

---
# Scraper
The **Scraper** component is just a **single function** that receives a list of urls to scrape and manages to select
the right **scraper object** for such url (based on its domain) or **raise an error** if it's not able to find any **matching scraper**.

## Example usage
The next examples will show you how to use the scraper to scrape a list of urls, handle a possible non-valid url
and filter out urls that may not be scrapable.
### Scraping multiple urls with the Manager object
The easiest way to scrape is using the manager object as follows:
```
import c4v.microscope as ms

# Creates the default manager
m = ms.Manager.from_default()

urls = [
    "https://primicia.com.ve/mas/servicios/siete-trucos-caseros-para-limpiar-la-plancha-de-ropa/",
    "https://primicia.com.ve/guayana/ciudad/suenan-con-urbanismo-en-core-8/"
]

# Output may depend on your internet connection and page availability
for result in m.scrape(urls):
    print(result.pretty_repr(max_content_len = 100))

```
### Scraping a single url
```
import c4v.microscope as ms

m = ms.Manager.from_default()

url = "https://primicia.com.ve/mas/servicios/siete-trucos-caseros-para-limpiar-la-plancha-de-ropa/"

# Output may depend on your internet connection and page availability
result = m.scrape(url)
print(result.pretty_repr(max_content_len = 100))
```

### Removing non-scrapable urls
Here we can see how to separate scrapable urls from non-scrapable ones. It may be helpful to know which urls can be processed
```
import c4v.microscope as ms

m = ms.Manager.from_default()

urls = [
    "https://primicia.com.ve",
    "https://elpitazo.net",
    "https://supernotscrapable.com"
]

assert m.split_non_scrapable(urls) == (urls[:2], urls[2:])
```
## Creation 
You can create a new scraper in order to support scraping for new sites. More details about this in ["creating a scraper"](./creating-a-scraper.md)
# Crawler
The easiest way to call a crawler to get more urls is to use the `microscope.Manager` object:

### Crawling N urls for a given site
```python
import c4v.microscope as ms

manager = ms.Manager.from_default()
# Crawls 100 urls using the crawler named "primicia"
manager.crawl("primicia", limit = 100)
```

But, how do you know which values could be supported by this command? 
### Getting possible crawlers
```python
import c4v.microscope as ms

manager = ms.Manager.from_default()

print(manager.available_crawlers())
# ['primicia', 'el_pitazo']
```
You can use this list to check if a given crawler is a valid one or not.

### Crawling new urls
You can get unknown urls by using the configured database by using the following code. You can choose to not save them if you prefer so.
```python
import c4v.microscope as ms

manager = ms.Manager.from_default()

# None of the retrieved urls are already in database, store them after retrieve
print(manager.crawl_new_urls_for(['primicia'], limit = 100))

# None of the retrieved urls are already in database, don't save them
print(manager.crawl_new_urls_for(['primicia'], limit = 100, save_to_db=False))
```

### Crawl and scrape
Perhaps you just want to crawl urls to scrape it afterwards, you can do easily following this example:
```python
import c4v.microscope as ms

manager = ms.Manager.from_default()

# Crawl, scrape, and save to db at the same time
manager.crawl_and_scrape_for(['primicia'], limit = 100)
```
## Creation
You can create a new crawler in order to support exploring new urls for new sites. More details about this in ["creating a crawler"](./creating-a-crawler.md)
# Persistency Manager
The persistency manager component helps you to specify how data is persisted.
There's multiple persistency managers, and users can even provide their own, but all of them should provide the same api. Right now, we have two specially important persistency managers:

1. `SQLiteManager` : A persistency manager to **store data in a local SQLite db**, used by the default `microscope.Manager` object configuration and the cli tool.
2. `BigQueryManager` : A persistency manager **that stores data in google cloud and firestore**. When a new instance comes from a crawling, **they're persisted to firestore** as long as its data is not filled by a scraper and a classifier. When all components run and the data is filled, **they're moved to Big Query**.

## Creation
You can create a new `Persistency Manager` object in order to support new ways of storing data. More details about this in ["creating a persistency manager"](./creating-a-persistency-manager.md)
# Experiment
An experiment is **a python script specifying a training run for a model class**, you can use them to create models and fast experimentation. Since they're python files, you can perform all the data manipulation you need before you run you're experiment. Right now we have the following models and experiments:

- `service_classification_experiment.py` : An experiment to train a multi label model into service type classification.

- `test_lang_model_train.py` : An experiment to train a model in a fill mask task in order to improve it's accuracy with a specific spanish dialect.

- `test_relevance_classifier.py` : An experiment to train a model to tell if an article is relevant or not.

# ExperimentFSManager
It's an object that will manage how experiments are saved. Usually, experiments are specified by a **branch** and an **experiment name**, in a file folder structure inside the c4v folder.


