"""
    An experiment represents a process ran by demand by a developer through scripting,
    which may need some special files and folders, and access to other
    components of the c4v library, the PersistencyManager class for example. 
    Also, experiments are provided with a folder for their necessary files.

    Experiments should provide a summary for results, and an Args Object to use 
    when they run.
"""
# Python imports
import pytz
import sys
import shutil
from pathlib     import Path
from datetime    import datetime
from typing      import Type
from dataclasses import dataclass, field

# Local imports
from c4v.config import settings

@dataclass
class BaseExperimentSummary:
    """
        Provide an output summary describing results for this experiment, 
        and an human readable representation
    """
    # Initialize to date when created
    date : datetime = field(default_factory=lambda: datetime.now(tz=pytz.utc))

    # Optional description
    description : str = None
    
    def __str__(self) -> str:
        # An human readable representation 
        output =  f"Date: {datetime.strftime( self.date, settings.date_format )}\n"
        output += f"Description: {self.description or '<no description>'}\n"

        return output

@dataclass
class BaseExperimentArguments:
    """
        Arguments to pass to an experiment run
    """
    pass

class BaseExperiment:
    """
        Inherit this class to create new experiments. Essentially, an experiment
        represents some parametrizable process that perform some actions and outputs
        some results. It may use some resources as database managers, scrapers and 
        datasets.
    """
    def __init__(self, base_folder : str = None) -> None:
        
        self._base_folder = base_folder

    @property
    def base_folder(self) -> str:
        """
            Folder to store files for this experiment if necessary
        """
        self._base_folder

    @base_folder.setter
    def base_folder(self, folder : str):
        if folder:
            assert Path(folder).exists(), f"Folder {folder} does not exists" # Checks if path exists
        self._base_folder = folder

    def run_experiment(self, args : BaseExperimentArguments) -> BaseExperimentSummary:
        """
            Run an experiment and return a summary for this experiment
        """
        raise NotImplementedError("Should implement run_experiment abstract function")
    
class ExperimentManager:
    """
        This class manages running experiments and storing results properly, 
        so experiments doesn't have to worry about where their files will be saved and 
        how their files are managed.
    """

    def __init__(self, 
                    experiment : BaseExperiment, 
                    branch_name : str, 
                    experiment_name : str, 
                    experiments_folder : str = None):

        self._branch_name     = branch_name
        self._experiment_name = experiment_name

        # Set up experiments 
        self._set_up_experiments_folder(experiments_folder)

        # Set up experiment folder
        #   Experiment expects the folder to be created beforehand, so create it if doesn't exists
        experiment.base_folder = self._get_or_create_path_to(self._get_experiment_folder_path())
        self._experiment       = experiment

    def _set_up_experiments_folder(self, custom_folder : str = None):
        """
            Set up experiment folder, if custom folder is provided, 
            use that folder and assume it does exists, otherwise, 
            create a default one
        """

        # if custom folder provided, just use it assuming it does exist        
        if custom_folder:
            self._experiments_folder = custom_folder
            return

        # Set up experiments folder
        experiments_folder_path = Path(settings.c4v_folder, "experiments/")

        # Create folder if not existd
        if not experiments_folder_path.exists():
            experiments_folder_path.mkdir(parents=True)

        self._experiments_folder = str(experiments_folder_path)            

    def _get_or_create_path_to(self, folder: str) -> str:
        """
            Get path to given folder and create it if doesn't exist
        """
        p = Path(folder)
        if not p.exists():
            p.mkdir(parents=True)

        return folder

    def _get_experiment_folder_path(self) -> str:
        """
            Get path to files for this experiment
        """
        return str(Path(
            self._experiments_folder, f"{self._branch_name}/{self._experiment_name}"
        ))

    def _write_summary(self, summary : BaseExperimentSummary):
        """
            Write summary to corresponding file and also to stdio
        """
        file_to_write = self._experiments_folder + "/summary.txt"

        with open(file_to_write, "w+") as f:
            summary_str = str(summary)
            print(summary_str, file=f) # Print to desired file
            print(summary_str)         # Print to stdio

    def run_experiment(self, args : BaseExperimentArguments, delete_after_run : bool = False) -> BaseExperimentSummary:
        """
            Run experiment, givin the given args to the configured experiment, storing its summary
            as a dump in a file in the experiment folder. Delete such folder at the end if requested so.
            Parameters:
                args : BaseExperimentArguments = Args to pass to the experiment when running
                delete_after_run : bool = Should delete experiment folder after run is finished
            Return:
                Experiment result as an Experiment Summary     
        """
        # Run experiment
        experiment = self._experiment
        summary = experiment.run_experiment(args)

        # Write summary
        self._write_summary(summary)

        # if nothing else to do, just return
        if not delete_after_run:
            return summary

        # Delete experiment files
        p = Path(self._get_experiment_folder_path())

        try:
            shutil.rmtree(p)
        except IOError as e:
            print(f"Couldn't delete folder {str(p)}, error: {e}", file=sys.stderr)

        return summary

