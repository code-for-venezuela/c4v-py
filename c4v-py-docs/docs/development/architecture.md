# Architecture

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

!!! Warning
    The **classifier** should be more specific in the future, it should be able not only to differentiate between news talking 
    about public services or not, but also the kind of problem itself
---
## Scraper
TODO
## Crawler
TODO
## Persistency Manager
TODO
## Experiment
TODO
## ExperimentFSManager
TODO
