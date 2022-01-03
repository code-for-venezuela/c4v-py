"""
    State for the application and functions used to retrieve data needed to feed the 
    page. This is basically a wrapper over c4v-py
"""
# Local imports
from typing import Callable, List
import c4v.microscope as ms
from c4v.scraper.persistency_manager.big_query_persistency_manager import BigQueryManager
from c4v.microscope.metadata import Metadata
from c4v.config import PersistencyManagers, settings

# Python imports
from pathlib import Path
import os

# Third party imports
import pandas as pd


class App:
    """
    Main application object, mainly used to get the data to show in the front end
    """

    # Valid options for labels
    label_options: List[str] = [
        "ANY",
        "IRRELEVANTE",
        "PROBLEMA DEL SERVICIO",
        "NO LABEL",
    ]

    # Valid options for scraped
    scraped_options: List[str] = ["Any", "Yes", "No"]

    def __init__(self, manager : ms.Manager = None) -> None:

        self._manager = manager or ms.Manager.from_default()

    @property
    def manager(self) -> ms.Manager:
        """
        Manager currently in use
        """
        return self._manager

    def get_dashboard_data(
        self,
        max_rows: int = 100,
        max_content_len: int = 200,
        label: str = "ANY",
        scraped: str = "Any",
    ) -> pd.DataFrame:
        """
        Get data to show in the dashboard
        # Parameters:
            - max_rows : `int` = (optional) maximum amount of rows to display
            - max_content_len : `int` = (optional) maximum amount of chars to show in the content field which might be long
            - label : `str` = (optional) description of the label that every instance should have
            - scraped : `str` = (optional) if the instances should be scraped.
        """
        # Some sanity check
        assert label in App.label_options, f"invalid label: {label}"
        assert scraped in App.scraped_options, f"invalid scraped option: {scraped}"

        # get value depending on if the instances should be scraped
        opt_2_val = {"Any": None, "Yes": True, "No": False}

        query = self._manager.get_all(scraped=opt_2_val[scraped])

        # Add filtering
        if label == "NO LABEL":
            query = (x for x in query if not x.label)
        elif label != "ANY":
            query = (x for x in query if x.label and x.label.value == label)

        elems = []
        for d in query:

            # In case of null data:
            # Reformat enum fields
            if d.source:
                d.source = d.source.value
            if d.label:
                d.label = d.label.value
            # Default to empty string
            d.content = d.content or ""

            # truncate body if needed
            content_len = len(d.content)
            d.content = (
                d.content
                if content_len < max_content_len
                else d.content[:max_content_len] + "..."
            )
            elems.append(d)

            # break if gathered enough rows
            if len(elems) == max_rows:
                break
        return pd.DataFrame(elems)

    @property
    def available_branchs(self) -> List[str]:
        """
        List of available branches
        """
        # TODO add a function to classifier object to get available experiments and branches for an experiment
        experiments_path = Path(self._manager.local_files_path, "experiments")
        return [str(x.name) for x in experiments_path.glob("*")]

    def available_experiments_for_branch(self, branch_name: str) -> List[str]:
        """
        List of available experiments for a given branch. Raise an error if invalid branch name is provided.

        # Parameters
            - branch_name : `str` = branch whose experiments are to be retrieved
        # Return
            List of experiments corresponding to the given branch
        """
        assert branch_name in self.available_branchs

        # TODO add function to the classifier object to get experiments from branch
        experiments_path = Path(
            self._manager.local_files_path, "experiments", branch_name
        )

        return [str(x.name) for x in experiments_path.glob("*")]

    def experiment_summary(self, branch_name: str, experiment_name: str) -> str:
        """
        Summary for the given experiment defined by its branch name and experiment name. Might be None if
        no summary is found
        # Parameters
            - branch_name : `str ` = Branch name for experiment
            - experiment_name : `str ` = Experiment name for experiment
        # Return
            Summary for the given experiment, or None if not found
        """
        # Sanity check
        assert branch_name in self.available_branchs
        assert experiment_name in self.available_experiments_for_branch(branch_name)

        # TODO Crear funcion en c4v manager que traiga el summary
        summary_path = Path(
            self._manager.local_files_path,
            "experiments",
            branch_name,
            experiment_name,
            "summary.txt",
        )

        # Return None if not exists
        if not summary_path.exists():
            return None

        return summary_path.read_text()

    def classify(self, branch_name: str, experiment_name: str, limit: int = -1):
        """
        Run a classification process.
        # Parameters
            - branch_name : `str ` = Branch name for experiment
            - experiment_name : `str ` = Experiment name for experiment
            - limit  : `int` = Max ammount of rows to classify, provide a negative number for no limit
        """
        self._manager.run_pending_classification_from_experiment(
            branch_name, experiment_name, limit=limit
        )

    def crawl(
        self,
        crawlers_to_use: List[str],
        limit: int,
        progress_function: Callable[[List[str]], None],
    ):
        """
        Run a crawling process
        # Parameters
            - crawlers_to_use : `[str]` = names of the crawlers to use during this crawling process
            - limit : `int` = max amount of rows to store. If limit < 0, no limit is assumed
            - progress_function : `[str] -> None` = Function to call whenever a new bulk of urls is received, intended to be used to add a progress bar
        """
        self._manager.crawl_and_process_new_urls_for(
            crawlers_to_use,
            post_process=progress_function,
            limit=limit,
            save_to_db=True,
        )

    def scrape(self, limit: int) -> int:
        """
        Run a scraping process, scraping only pending rows up to the given limit
        # Parameters
            - limit : `int` = max amount of rows to scrape. If `limit < 0`, 'no limit' is assumed
        # Return
            Exit code for the scraping sub process, 0 if success, anything else for error
        """
        # No pude programar esto de forma que llamara a la función scrape del manejador principal, porque la arquitectura de scrapy
        # demanda estar en el el thread principal, cosa que no pasa con streamlit, así que esto fue lo mejor que pude hacer
        assert isinstance(limit, int)
        return os.system(f"c4v scrape --limit {limit}")

    def upload_model_of_type(self, experiment: str, branch: str, type: str):
        """
        Upload a classifier model
        """
        self.manager.upload_model(experiment, branch, type)

    def download_model_of_type(self, path: str, type: str):
        """
        Download the model of type 'type' to path 'path'
        """
        self.manager.download_model_to_directory(path, type)

    @classmethod
    def cloud_backend():
        """
            Create a cloud backed version of this object
        """
        return CloudApp()

class CloudApp(App):
    """
        App implementation based on cloud actions. Will override 
        operations like scraping, crawling and classifying to perform a cloud request 
    """
    def __init__(self) -> None:

        metadata = Metadata(persistency_manager=PersistencyManagers.GCLOUD.value)
        manager = ms.Manager.from_default(metadata=metadata)
        super().__init__(manager=manager)

    def classify(self, branch_name: str, experiment_name: str, limit: int = -1):
        raise NotImplementedError()
        return super().classify(branch_name, experiment_name, limit=limit)
    
    def scrape(self, limit: int) -> int:
        raise NotImplementedError()
        return super().scrape(limit)
    
    def crawl(self, crawlers_to_use: List[str], limit: int, progress_function: Callable[[List[str]], None]):
        raise NotImplementedError()
        return super().crawl(crawlers_to_use, limit, progress_function)