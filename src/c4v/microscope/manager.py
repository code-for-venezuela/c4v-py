"""
    This file exposes the main API for this library, the microscope Manager
"""
# Local imports 
from c4v.scraper.persistency_manager.base_persistency_manager   import BasePersistencyManager
from c4v.scraper.persistency_manager.sqlite_storage_manager     import SqliteManager
from c4v.scraper.scraped_data_classes.scraped_data              import ScrapedData
from c4v.scraper.scraper                                        import bulk_scrape, _get_scraper_from_url
from c4v.scraper.settings                                       import INSTALLED_CRAWLERS
from c4v.classifier.classifier                                  import ClassifierExperiment

# Python imports
from typing import Dict, List, Iterable, Callable, Tuple, Any
import sys

class Manager:
    """
        This object encapsulates shared behavior between our multiple components,
        allowing easy access to common operations
    """

    def __init__(self, persistency_manager: BasePersistencyManager):
        self._persistency_manager = persistency_manager

    def get_bulk_data_for(self, urls: List[str], should_scrape: bool = True) -> List[ScrapedData]:
        """
            Retrieve scraped data for given url set if scrapable
            Parameters:
                urls : [str] = urls whose data is to be retrieved. If not available yet, then scrape it if requested so
                should_scrape : bool = if should scrape non-existent urls
        """
        # just a shortcut
        db = self._persistency_manager

        # Separate scraped urls from non scraped
        not_scraped = db.filter_scraped_urls(urls)

        # Scrape missing instances if necessary
        if should_scrape and not_scraped:
            items = bulk_scrape(not_scraped)
            db.save(items)

        # Convert to set to speed up lookup
        urls = set(urls)
        return [sd for sd in db.get_all() if sd.url in urls]

    def get_data_for(self, url: str, should_scrape: bool = True) -> ScrapedData:
        """
            Get data for this url if stored and scrapable. May return none if could not
            find data for this url
            Parameters:
                url : str = url to be scraped
                should_scrape : bool = if should scrape this url if not available in db
        """
        data = self.get_bulk_data_for([url], should_scrape)
        return data[0] if data else None

    def scrape_pending(self, limit: int = -1):
        """
            Update DB by scraping rows with no scraped data, just the url
            Parameters:
                limit : int = how much measurements to scrape, set a negative number for no limit
        """

        db = self._persistency_manager

        scrape_urls = [d.url for d in db.get_all(limit=limit, scraped=False)]

        scraped = bulk_scrape(scrape_urls)

        db.save(scraped)

    def get_all(self, limit : int = -1, scraped : bool = None) -> Iterable[ScrapedData]:
        """
            Return all ScrapedData instances available, up to "limit" rows. If scraped = true, then
            return only scraped instances. If scraped = false, return only non scraped. Otherwise, return anything
            Parameters:
                limit : int = max ammount of rows to return. If negative, there's no limit of rows
                scraped : bool = If instances should be scraped or not. If true, all of them should be scraped. If false, none of 
                                them should be scraped. If None, there's no restriction
            Return:
                An iterator returning available rows
        """ 
        return self._persistency_manager.get_all(limit, scraped)

    def crawl_new_urls_for(self, crawler_names: List[str] = None, post_process : Callable[[List[str]], None] = None, limit = -1):
        """
            Crawl for new urls using the given crawlers only
            Parameters:
                crawler_names : [str] = names of crawlers to be ran when this function is called. If no list is passed, then 
                                        all crawlers will be used
                post_process : ([str]) -> None = Function to call over new elements as they come
                limit        : int = Max amount of urls to save
        """
        db = self._persistency_manager
        class Counter:
            def __init__(self):
                self.count = 0
                
            def add(self, x : int):
                self.count += x

        counter = Counter()

        # Function to process urls as they come
        def save_urls(urls: List[str]):
            urls = db.filter_scraped_urls(urls)
            datas = [ScrapedData(url=url) for url in urls]
            db.save(datas)

            # Update how much elements have beed added so far
            counter.add(len(datas))

            # Call any callback function 
            if post_process:
                post_process(urls)

        # Function to tell if the crawling process should stop
        def should_stop() -> bool:
            return counter.count >= limit

        # Names for installed crawlers
        crawlers = [c.name for c in INSTALLED_CRAWLERS]

        # if no list provided, default to every crawler
        if crawler_names == None:
            crawler_names = crawlers

        not_registered = [name for name in crawler_names if name not in crawlers]

        # Report warning if there's some non registered crawlers
        if not_registered:
            print(
                "WARNING: some names in given name list don't correspond to any registered crawler.",
                file=sys.stderr,
            )
            print(
                "Unregistered crawler names: \n"
                + "\n".join([f"\t* {name}" for name in not_registered])
            )

        # Instantiate crawlers to use
        crawlers_to_run = [
            crawler() for crawler in INSTALLED_CRAWLERS if crawler.name in crawler_names
        ]

        # crawl for every crawler
        for crawler in crawlers_to_run:
            crawler.crawl_and_process_urls(save_urls, should_stop)

    def split_non_scrapable(self, urls : List[str]) -> Tuple[List[str], List[str]]:
        """
            splits url list in two list, one having a list of scrapable urls, 
            and another one with only non-scrapable
            Parameters:
                urls : [str] = List of urls to be split
            Return:
                A tuple with two lists, the first one with only scrapable urls from input list,
                the second one with non-scrapable ones
        """
        scrapable, non_scrapable = [], []

        for url in urls:
            try:
                _get_scraper_from_url(url)
                scrapable.append(url)
            except ValueError:
                non_scrapable.append(url)

        return scrapable, non_scrapable

    @classmethod
    def from_local_sqlite_db(cls, db_path : str):
        """
            Create a new instance using an SQLite local db
        """
        db = SqliteManager(db_path)
        return cls(db)

    def run_classification_from_experiment(self, branch : str, experiment : str, data : List[ScrapedData]) -> Dict[str, Dict[str, Any]]:
        """
            Classify given data instance list, returning its metrics
            Parameters:
                branch : str = branch of model to use
                experiment : str = experiment name storing model
                data : [ScrapedData] = Instance to be classified
            Return:
                A dict from urls to classification output
        """
        classifier = ClassifierExperiment(branch, experiment)
        classified = { d.url : classifier.classify(d) for d in data }

        return classified
        