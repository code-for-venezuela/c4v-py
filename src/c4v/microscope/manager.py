"""
    This file exposes the main API for this library, the microscope Manager
"""
# Local imports
from c4v.ray.ray_crawler import ray_crawl_and_process_urls
from c4v.scraper.persistency_manager.base_persistency_manager import (
    BasePersistencyManager,
)
from c4v.scraper.persistency_manager.sqlite_storage_manager import SqliteManager
from c4v.scraper.scraped_data_classes.scraped_data          import ScrapedData
from c4v.scraper.scraper                                    import bulk_scrape, _get_scraper_from_url
from c4v.scraper.settings                                   import INSTALLED_CRAWLERS
from c4v.classifier.classifier_experiment                   import ClassifierExperiment
from c4v.classifier.classifier                              import Classifier

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

    def get_bulk_data_for(
        self, urls: List[str], should_scrape: bool = True,
        use_ray : bool = False
    ) -> List[ScrapedData]:
        """
            Retrieve scraped data for given url set if scrapable
            Parameters:
                urls : [str] = urls whose data is to be retrieved. If not available yet, then scrape it if requested so
                should_scrape : bool = if should scrape non-existent urls
                use_ray : bool = If should use ray to scrape in a distributed manner
            Return:
                List of scraped data instances
        """
        # just a shortcut
        db = self._persistency_manager

        # Separate scraped urls from non scraped
        not_scraped = db.filter_scraped_urls(urls)

        # Scrape missing instances if necessary
        if should_scrape and not_scraped:
            if use_ray:
                # import ray here as it may not be installed in development profiles
                try:
                    from c4v.ray.ray_scraper import ray_scrape
                except ImportError as e:
                    raise ImportError(f"Could not import ray implementations, maybe you installed the wrong profile?. Error: {e}")
                items = ray_scrape(not_scraped)
            else:
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

    def get_all(self, limit: int = -1, scraped: bool = None) -> Iterable[ScrapedData]:
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

    def crawl_new_urls_for(
        self,
        crawler_names: List[str] = None,
        post_process: Callable[[List[str]], None] = None,
        limit : int =-1,
        use_ray : bool = False
    ):
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

            def add(self, x: int):
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

        # if use ray, then:
        if use_ray:
            # import ray here as it may not be installed in development profiles
            try:
                from c4v.ray.ray_crawler import ray_crawl
            except ImportError as e:
                raise ImportError(f"Could not import ray implementations, maybe you installed the wrong profile?. Error: {e}")

            ray_crawl_and_process_urls(crawler_names, save_urls, should_stop)


        # Instantiate crawlers to use
        crawlers_to_run = [
            crawler_class for crawler_class in INSTALLED_CRAWLERS if crawler_class.name in crawler_names
        ]

        # crawl for every crawler
        for crawler_class in crawlers_to_run:
            crawler = crawler_class()

            crawler.crawl_and_process_urls(save_urls, should_stop)

    def split_non_scrapable(self, urls: List[str]) -> Tuple[List[str], List[str]]:
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
    def from_local_sqlite_db(cls, db_path: str):
        """
            Create a new instance using an SQLite local db
        """
        db = SqliteManager(db_path)
        return cls(db)

    def run_classification_from_experiment(
        self, branch: str, experiment: str, data: List[ScrapedData]
    ) -> Dict[str, Dict[str, Any]]:
        """
            Classify given data instance list, returning its metrics
            Parameters:
                branch : str = branch of model to use
                experiment : str = experiment name storing model
                data : [ScrapedData] = Instance to be classified
            Return:
                A dict from urls to classification output
        """

        classifier_experiment = ClassifierExperiment.from_branch_and_experiment(
            branch, experiment
        )

        classified = {d.url: classifier_experiment.classify(d) for d in data}

        return classified

    def explain_for_experiment(
        self,
        branch: str,
        experiment: str,
        sentence: str,
        html_file: str = None,
        additional_label: str = None,
    ) -> Dict[str, Any]:
        """
            Explain a sentence using the given branch and experiment
            Parameters:
                branch : str = branch name of model to use
                experiment : str = experiment name storing model
                sentence : str = sentence to explain
                additional_label : str = Label to include in expalantion. If the predicted 
                                         label is different from this one, then explain how 
                                         much this label was contributing to its corresponding value, 
                                         ignored if not provided
            Return:
                Dict with explaination data
        """

        classifier_experiment = ClassifierExperiment.from_branch_and_experiment(
            branch, experiment
        )

        return classifier_experiment.explain(
            sentence, html_file, additional_label=additional_label
        )

    def get_classifier_labels(self) -> List[str]:
        """
            Get list of possible labels for classifier
            Return:
                List with possible output labels for the classifier
        """
        return Classifier.get_labels()
