"""
    This file exposes the main API for this library, the microscope Manager
"""
# Local imports
from c4v.scraper.persistency_manager.base_persistency_manager import (
    BasePersistencyManager
)
from c4v.scraper.persistency_manager.sqlite_storage_manager import SqliteManager
from c4v.scraper.scraped_data_classes.scraped_data import ScrapedData, Sources, RelevanceClassificationLabels
from c4v.scraper.scraper import bulk_scrape, _get_scraper_from_url
from c4v.scraper.settings import INSTALLED_CRAWLERS
from c4v.config import settings
from c4v.microscope.metadata import Metadata
import c4v.microscope.utils as utils
from c4v.config import PersistencyManagers, settings


# Python imports
from typing import Dict, List, Iterable, Callable, Tuple, Any, Union
from pathlib import Path
import sys


class Manager:
    """
        This object encapsulates shared behavior between our multiple components,
        allowing easy access to common operations.
        Parameters:
            persistency_manager : BasePersistencyManager = persistency manager object to use when accessing persistent data storage
            metadata : Metadata = Persistent configuration data
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

    def get_bulk_data_for(
        self, urls: List[str], should_scrape: bool = True, save: bool = True
    ) -> List[ScrapedData]:
        """
            Retrieve scraped data for given url set if scrapable
            Parameters:
                urls : [str] = urls whose data is to be retrieved. If not available yet, then scrape it if requested so
                should_scrape : bool = (optional) if should scrape non-existent urls
                save : bool = (optional) if should save new data when scraped for the first time
        """
        # just a shortcut
        db = self._persistency_manager

        # Separate scraped urls from non scraped
        not_scraped = db.filter_scraped_urls(urls)

        # Scrape missing instances if necessary
        if should_scrape and not_scraped:
            items = bulk_scrape(not_scraped)
            if save:
                db.save(items)  # save if requested to

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
        limit=-1,
        save_to_db : bool = True
    ):
        """
            Crawl for new urls using the given crawlers only
            Parameters:
                crawler_names : [str] = names of crawlers to be ran when this function is called. If no list is passed, then 
                                        all crawlers will be used
                post_process : ([str]) -> None = Function to call over new elements as they come
                limit        : int = Max amount of urls to save, -1 when no limit 
        """
        db = self._persistency_manager
        limit = limit if limit >= 0 else sys.maxsize

        class Counter:
            def __init__(self):
                self.count = 0

            def add(self, x: int):
                self.count += x

        counter = Counter()

        # Function to process urls as they come
        def save_urls(urls: List[str]):
            if db:
                # Filter already known urls and take at the must the necessary ones to fill the required size
                urls = db.filter_known_urls(urls)[: limit - counter.count]
                datas = [ScrapedData(url=url, source=Sources.SCRAPING) for url in urls]

                if save_to_db: db.save(datas)

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

        # Instantiate crawlers to use
        crawlers_to_run = [
            crawler() for crawler in INSTALLED_CRAWLERS if crawler.name in crawler_names
        ]

        # crawl for every crawler
        for crawler in crawlers_to_run:
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
        cls,
        db_path: str = None,
        metadata: Union[str, Metadata] = None,
        local_files_path: str = None,
    ):
        """
            Create a Manager instance using files from the default `C4V_FOLDER`
        """

        # Set up metadata
        if isinstance(metadata, str) or metadata == None:
            metadata = Metadata.from_json(metadata) if metadata else Metadata()
        elif not isinstance(metadata, Metadata):
            raise TypeError("Unsupported type for metadata")

        # Set up db
        if metadata.persistency_manager == PersistencyManagers.SQLITE.value:
            db = SqliteManager(
                db_path
                or str(
                    Path(
                        local_files_path or settings.c4v_folder,
                        settings.local_sqlite_db_name,
                    )
                )
            )
        elif metadata.persistency_manager == PersistencyManagers.USER.value:
            if not metadata.user_persistency_manager_module:
                raise ValueError(
                    f"Requested to create persistency manager from user defined class, but its module name wasn't provided"
                )
            elif not metadata.user_persistency_manager_path:
                raise ValueError(
                    f"Requested to create persistency manager from user defined class, but its path wasn't provided"
                )
            db = utils._load_user_manager(
                metadata.user_persistency_manager_module,
                metadata.user_persistency_manager_path,
            )
        else:
            raise NotImplementedError(
                f"Not implemented default db creation for db type: {settings.persistency_manager}"
            )

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
        from c4v.classifier.classifier_experiment import ClassifierExperiment

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
            x for x in self.persistency_manager.get_all(scraped=True) if not x.label_relevance
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
                html_file : str = where to store results in html. If not provided, they wont be stored
            Return:
                Dict with explaination data
        """

        from c4v.classifier.classifier_experiment import ClassifierExperiment

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
        from c4v.classifier.classifier import Classifier
        raise NotImplementedError("Get classifier labels should be reimplemented")
        return Classifier.get_labels()

    def should_retrain_base_lang_model(
        self,
        lang_model, #: LanguageModel,
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

        from c4v.classifier.language_model.language_model import LanguageModel

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
