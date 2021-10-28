"""
    This file exposes the main API for this library, the microscope Manager
"""
# Local imports
from c4v.scraper.persistency_manager.base_persistency_manager import (
    BasePersistencyManager,
)
from c4v.scraper.persistency_manager.sqlite_storage_manager import SqliteManager
from c4v.scraper.scraped_data_classes.scraped_data import ScrapedData, Sources
from c4v.scraper.scraper import bulk_scrape, _get_scraper_from_url, scrape
from c4v.scraper.settings import INSTALLED_CRAWLERS, INSTALLED_SCRAPERS, SUPPORTED_DOMAINS
from c4v.classifier.classifier_experiment import ClassifierExperiment
from c4v.classifier.classifier import Classifier
from c4v.classifier.language_model.language_model import LanguageModel
from c4v.config import PersistencyManagers, settings
from c4v.microscope.metadata import Metadata
import c4v.microscope.utils as utils

# Python imports
import os
from typing import Dict, List, Iterable, Callable, Tuple, Any, Type, Union
from pathlib import Path
import sys

# Configs
from c4v.config import settings

DEFAULT_DB = settings.local_sqlite_db or os.path.join(
    settings.c4v_folder, settings.local_sqlite_db_name
)

class Manager:
    """
        This object encapsulates shared behavior between our multiple components,
        allowing easy access to common operations. Note that this operations are intended to work 
        with a local db, so most of them will save data to some kind of storage. If no local db is 
        provided, some operations may not work
    """

    def __init__(
        self,
        persistency_manager: BasePersistencyManager,
        metadata: Metadata,
        local_files_path: str = settings.c4v_folder,
    ):
        self._persistency_manager = persistency_manager
        self._metadata = metadata
        self._local_files_path = local_files_path

    @property
    def local_files_path(self) -> str:
        """
            Path where files (.c4v folder) will be saved
        """
        return self._local_files_path

    @local_files_path.setter
    def local_files_path(self, new_path: str):
        self._local_files_path = new_path

    @property
    def persistency_manager(self) -> BasePersistencyManager:
        return self._persistency_manager

    def scrape(self, urls : Union[List[str], str] ) -> Union[List[ScrapedData], ScrapedData]:
        """
            Scrape the given url or set of urls and return its results
        """
        if not urls: # if nothing to process, return None
            return None 
        elif isinstance(urls, list) and urls and isinstance(urls[0], str):
            return bulk_scrape(urls)
        elif isinstance(urls, str):
            return scrape(urls)
        
        raise TypeError(f"Expected argument should be a list of urls as strings or a single string url. Given {type(urls)}")

    def crawl_and_scrape_for(self, crawler_names : List[str] = None, limit : int = -1, save_to_db = True) -> List[ScrapedData]:
        """
            Crawl and scrape data for the given list of crawlers
            Parameters:
                crawler_names : [str] = List of crawlers to use, defaults to all crawlers
                limit : int = Maximum ammount of urls to scrape, set to negative number for no limit (no recommended)
            Return:
                List of scraped data    
        """
        urls = self.crawl_new_urls_for(crawler_names=crawler_names, limit=limit, save_to_db=save_to_db)
        data = self.get_bulk_data_for(urls, save_to_db=save_to_db)
        return data

    def get_bulk_data_for(
        self, urls: List[str], should_scrape: bool = True, save_to_db = True
    ) -> List[ScrapedData]:
        """
            Retrieve scraped data for given url set if scrapable
            Parameters:
                urls : [str] = urls whose data is to be retrieved. If not available yet, then scrape it if requested so
                should_scrape : bool = if should scrape non-existent urls
                save_to_db : bool = if should save to db
            Return:
                List of gathered data, 
        """
        # just a shortcut
        db = self._persistency_manager

        # Separate scraped urls from non scraped
        not_scraped = db.filter_scraped_urls(urls) if db else urls
        
        # Scrape missing instances if necessary
        items = []
        if should_scrape and not_scraped:
            items = bulk_scrape(not_scraped)
            if save_to_db and db:
                db.save(items)

        urls = set(urls)
        return [sd for sd in db.get_all() if sd.url in urls] if db else items

    def get_data_for(self, url: str, should_scrape: bool = True) -> ScrapedData:
        """
            Get data for this url if stored or scrapable. May return none if could not
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
        if not self._persistency_manager:
            raise AttributeError("Storage scheme not configured for microscope Manager instance, provide a db scheme to use this function")

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
        if not self._persistency_manager:
            raise AttributeError("Storage scheme not configured for microscope Manager instance, provide a db scheme to use this function")

        return self._persistency_manager.get_all(limit, scraped)

    def crawl_and_process_new_urls_for(
        self,
        crawler_names: List[str] = None,
        post_process: Callable[[List[str]], None] = None,
        limit=-1,
        save_to_db = True
    ):
        """
            Crawl for new urls using the given crawlers only
            Parameters:
                crawler_names : [str] = names of crawlers to be ran when this function is called. If no list is provided, then 
                                        all crawlers will be used
                post_process : ([str]) -> None = Function to call over new elements as they come
                limit        : int = Max amount of urls to save
                save_to_db : bool = If should save to local
        """
        db = self._persistency_manager
        limit = limit if limit >= 0 else sys.maxsize

        # set up limit if necessary
        limit = limit if limit >= 0 else sys.maxsize

        class Counter:
            def __init__(self):
                self.count = 0

            def add(self, x: int):
                self.count += x

        counter = Counter()

        # Function to process urls as they come
        def save_urls(urls: List[str]):

            urls = urls[:limit - counter.count]

            if db:
                urls = db.filter_scraped_urls(urls)
                datas = [ScrapedData(url=url) for url in urls]

                if save_to_db: db.save(datas)

            # Update how much elements have beed added so far
            counter.add(len(urls))

            # Call any callback function
            if post_process:
                post_process(urls)

        # Function to tell if the crawling process should stop
        def should_stop() -> bool:
            return counter.count >= limit

        # if no list provided, default to every crawler
        if crawler_names == None:
            crawler_names = [c.name for c in INSTALLED_CRAWLERS]

        # Instantiate crawlers to use
        crawlers_to_run = [
            crawler() for crawler in INSTALLED_CRAWLERS if crawler.name in crawler_names
        ]

        # Warns the user if no crawler could be found
        if not crawlers_to_run:
            given_crawler_list = "\n".join([f"\t* {c}" for c in crawler_names] or ["\t<No crawlers>"])
            actual_crawler_list = "\n".join([f"\t* {c.name}" for c in INSTALLED_CRAWLERS] or ["\t<No crawlers>"])
            print(f"[Warning] No crawler available to run. Given crawler list: \n{given_crawler_list}\n Available crawlers: \n{actual_crawler_list}", file=sys.stderr)

        # crawl for every crawler
        for crawler in crawlers_to_run:
            crawler.crawl_and_process_urls(save_urls, should_stop)

    def crawl_new_urls_for( self,
        crawler_names: List[str] = None,
        limit=-1,
        save_to_db : bool = True
    ):
        """
            Crawl new urls for the given list of crawlers:
            Parameters:
                crawler_names : [str] = list of names of scrapers to use when crawling
        """
        items = []
        
        def collect_urls(urls : List[str]):
            items.extend(urls)

        self.crawl_and_process_new_urls_for(crawler_names=crawler_names, limit=limit, post_process=collect_urls, save_to_db=save_to_db)

        return items

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
    def from_local_sqlite_db(cls, db_path: str, metadata: Union[str, Metadata]):
        """
            Create a new instance using an SQLite local db
            metadata : str | Metadata = Path to file with metadata or metadata instance itself
        """
        if isinstance(metadata, Metadata):
            pass  # Everything ok, just keep going
        elif isinstance(metadata, str):
            # If string is provided, interpret it as a filepath
            metadata = Metadata.from_json(metadata)
        else:
            raise TypeError(
                f"metadata field should be path to metadata files or metadata instance. Provided object's type: {type(metadata)}"
            )

        db = SqliteManager(db_path)
        return cls(db, metadata)

    @classmethod
    def from_default(
        cls, db_path: str = None, metadata: str = None, local_files_path: str = None, db : BasePersistencyManager = None
    ):
        """
            Create a Manager instance using files from the default `C4V_FOLDER`.
            Parameters:
                - db_path : str = Path to a sqlite db file to use as database.
                - metadata : str = Path to a valid metadata json file 
                - local_files_path : str = path to the .c4v folder to use for local files
                - db : BasePersistencyManager = persistency manager object to use as db manager. When provided, it overrides the db_path 
                                                parameter

        """

        # Set up db
        if db: # if custom db is provided, use it instead of the default one
            pass
        elif settings.persistency_manager == PersistencyManagers.SQLITE.value:
            db = SqliteManager(
                db_path
                or str(
                    Path(
                        local_files_path or settings.c4v_folder,
                        settings.local_sqlite_db_name,
                    )
                )
            )
        elif settings.persistency_manager == PersistencyManagers.USER.value:
            if not settings.user_persistency_manager_module:
                raise ValueError(f"Requested to create persistency manager from user defined class, but its module name wasn't provided")
            elif not settings.user_persistency_manager_path:
                raise ValueError(f"Requested to create persistency manager from user defined class, but its path wasn't provided")

            db = utils._load_user_manager(
                settings.user_persistency_manager_module, 
                settings.user_persistency_manager_path
            )
        else:
            raise NotImplementedError(f"Not implemented default db creation for db type: {settings.persistency_manager}")

        # Set up metadata
        metadata = Metadata.from_json(metadata) if metadata else Metadata()

        return cls(db, metadata, local_files_path or settings.c4v_folder)

    def run_classification_from_experiment(
        self, branch: str, experiment: str, data: List[ScrapedData]
    ) -> List[Dict[str, Any]]:
        """
            Classify given data instance list, returning its metrics
            Parameters:
                branch : str = branch name of model to use
                experiment : str = experiment name storing model
                data : [ScrapedData] = Instance to be classified
            Return:
                A List of dicts with the resulting scraped data correctly labelled
                and its corresponding scores tensor for each possible label. Available fields:
                    + data : ScrapedData = resulting data instance after classification
                    + scores : torch.Tensor = Scores for each label returned by the classifier
        """

        classifier_experiment = ClassifierExperiment.from_branch_and_experiment(
            branch, experiment
        )

        return classifier_experiment.classify(data)

    def run_pending_classification_from_experiment(
        self, branch: str, experiment: str, save: bool = True, limit: int = -1
    ) -> List[Dict[str, Any]]:
        """
            Classify data pending for classification in local db, returning the obtained results and saving it 
            to database. 
            Parameters:
                branch : str = branch name
                experiment : str = experiment name
                save : bool = if should store results in db
                limit : maximum number of rows to classify
            Return:
                A List of dicts with the resulting scraped data correctly labelled
                and its corresponding scores tensor for each possible label. Available fields:
                    + data : ScrapedData = resulting data instance after classification
                    + scores : torch.Tensor = Scores for each label returned by the classifier
        """
        # Parse limit
        limit = limit if limit >= 0 else sys.maxsize

        # Request at the most "limit" instances
        data = list(
            x for x in self.persistency_manager.get_all(scraped=True) if not x.label
        )[:limit]

        # classify
        results = self.run_classification_from_experiment(branch, experiment, data)

        # Save if requested so
        if save:
            self.persistency_manager.save((x["data"] for x in results))

        return results

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

    @staticmethod
    def get_available_crawlers() -> List[str]:
        """
            List of usable names of crawlers
        """
        return [c.name for c in INSTALLED_CRAWLERS]

    @classmethod
    def scrapable_domains() -> List[str]:
        """
            List of scrapable domains, it may not match with the crawlable sites
        """
        return SUPPORTED_DOMAINS

    def should_retrain_base_lang_model(
        self,
        lang_model: LanguageModel,
        db_manager: BasePersistencyManager = None,
        eval_dataset_size: int = 250,
        min_loss: float = settings.default_lang_model_min_loss,
        should_retrain_fn: Callable[[float], bool] = None,
        fields: List[str] = ["title"],
    ) -> bool:
        """
            If should retrain a base language model based on its loss on the given dataset. If a persistency manager is provided, 
            use it instead of the configured one
            Parameters:
                lang_model : LanguageModel = Language Model to reevaluate
                db_manager : BasePersistency manager = (optional) Persistency manager object to retrieve data to use when evaluating
                eval_dataset_size : int = (optional) Size of the dataset to use when evaluating the model
                min_loss : float = (optional) Minimum acceptable loss, if the computed loss is greater than this treshold, returns true. Otherwise returns false
                should_retrain_fn : (float) -> bool = (optional) Function to check if, based on the loss of the model, it should be retrained. 
                                                    Should receive the loss and return a boolean telling if it should be retrained. 
                                                    Providing this argument will override min_loss argument
                fields : [str] = (optional) fields from scraped data instances to use during evaluation
            Return:
                If the given model should be retrained
        """
        # Sanity check
        assert eval_dataset_size > 0, "Eval dataset size should be a possitive number"
        assert min_loss >= 0, "min loss should be non negative"

        # set up retrain function
        should_retrain_fn = should_retrain_fn or (lambda x: x > min_loss)

        # Set up db_manager
        db_manager = db_manager or self._persistency_manager

        # TODO should order by newer, but can't do it right now as we can't ensure
        # news dates can be parsed into datetime objects
        scraped_data = list(db_manager.get_all(limit=eval_dataset_size, scraped=True))
        if not scraped_data:
            raise ValueError(
                "No ScrapedData available for evaluation. You can get more data by using the `c4v scrawl` and `c4v scrape` functions, or using the microscope.Manager object"
            )

        # Set up dataset
        ds = LanguageModel.to_pt_dataset(
            lang_model.create_dataset_from_scraped_data(scraped_data, fields)
        )
        del scraped_data

        # Compute loss
        loss = lang_model.eval_accuracy(ds)
        return should_retrain_fn(loss)

