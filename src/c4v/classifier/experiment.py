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
from pathlib import Path
from datetime import datetime
from typing import List, Type
from dataclasses import dataclass, field


# Local imports
from c4v.config import settings


@dataclass
class BaseExperimentSummary:
    """
        Provide an output summary describing results for this experiment, 
        and an human readable representation.
        
        Note that in order to inherit this class, you must provide default 
        values for every field (required by dataclasses). If you want
        to mark some fields as mandatory, you can do so in the __post_init__
        method.
    """

    # Initialize to date when created
    date: datetime = field(default_factory=lambda: datetime.now(tz=pytz.utc))

    # Optional description
    description: str = None

    def __str__(self) -> str:
        # An human readable representation
        output = f"Date: {datetime.strftime( self.date, settings.date_format )}\n"
        output += f"Description: {self.description or '<no description>'}\n"

        return output


@dataclass
class BaseExperimentArguments:
    """
        Arguments to pass to an experiment run
    """

    pass


class ExperimentFSManager:
    """
        This class manages files for experiments and storing results properly, 
        so experiments doesn't have to worry about where their files will be saved and 
        how their files are managed.

        Experiments are (usually) stored in the 'experiments' folder, under the c4v folder,
        typically in: $HOME/.c4v/experiments.

        Every Experiment has a branch name and an experiment name, the folder provided to every 
        experiment is <experiments_folder>/<branch_name>/<experiment_name>
    """

    _experiments_folder = settings.experiments_dir

    def __init__(
        self, branch_name: str, experiment_name: str, experiments_folder: str = None
    ):

        self._branch_name = branch_name
        self._experiment_name = experiment_name

        # Set up experiments
        self._set_up_experiments_folder(experiments_folder)

    def _set_up_experiments_folder(self, custom_folder: str = None):
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

        # Create folder if not exists
        if not experiments_folder_path.exists():
            experiments_folder_path.mkdir(parents=True)

        self._experiments_folder = str(experiments_folder_path)

        # Set up experiments path
        experiment_content_path = Path(self.experiment_content_folder)
        if not experiment_content_path.exists():
            experiment_content_path.mkdir(parents=True)

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
        return str(
            Path(
                self._experiments_folder, f"{self._branch_name}/{self._experiment_name}"
            )
        )

    def write_summary(self, summary: BaseExperimentSummary):
        """
            Write summary to corresponding file
        """
        file_to_write = self.experiment_content_folder + "/summary.txt"

        with open(file_to_write, "w+") as f:
            print(f"Writing summary to file: {file_to_write}")
            summary_str = str(summary)
            print(summary_str, file=f)  # Print to desired file

    def delete_experiment_content_folder(self):
        """
            Delete experiment content folder
        """
        p = Path(self.experiment_content_folder)

        try:
            shutil.rmtree(p)
        except IOError as e:
            print(f"Couldn't delete folder {str(p)}, error: {e}", file=sys.stderr)

    @property
    def experiment_content_folder(self) -> str:
        """
            Path to folder where experiment content is stored, for example:
                $HOME/.c4v/experiments/<branch_name>/<experiment_name>
        """
        return self._get_experiment_folder_path()

    @property
    def experiments_folder(self) -> str:
        """
            Path to folder where experiments are stored, for example:
                $HOME/.c4v/experiments
        """
        return self._experiments_folder


class BaseExperiment:
    """
        Inherit this class to create new experiments. Essentially, an experiment
        represents some parametrizable process that perform some actions and outputs
        some results. It may use some resources as database managers, scrapers and 
        datasets.

        In order to create an Experiment, you have to inherit this class and implement the following
        functions:
            * experiment_to_run
    """

    def __init__(self, experiment_fs_manager: ExperimentFSManager) -> None:
        self._experiment_fs_manager = experiment_fs_manager

    @property
    def base_folder(self) -> str:
        """
            Folder to store files for this experiment if necessary
        """
        return self._experiment_fs_manager.experiment_content_folder

    def experiment_to_run(self, args: BaseExperimentArguments) -> BaseExperimentSummary:
        """
            Experiment to be ran when run_experiment function is called
            Parameters:
                args : BaseExperimentArguments = Args for the experiment to run
            Return:
                BaseExperimentSummary object summarizing results for this experiment run
        """
        raise NotImplementedError(
            "Should implement experiment_to_ron abstract function"
        )

    def run_experiment(
        self, args: BaseExperimentArguments, delete_after_run: bool = False
    ) -> BaseExperimentSummary:
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
        summary = self.experiment_to_run(args)

        # Write summary
        self._experiment_fs_manager.write_summary(summary)

        # if nothing else to do, just return
        if not delete_after_run:
            return summary

        # Delete experiment files
        self._experiment_fs_manager.delete_experiment_content_folder()
        return summary

    @classmethod
    def from_branch_and_experiment(cls, branch_name: str, experiment_name: str):
        fs_manager = ExperimentFSManager(branch_name, experiment_name)
        return cls(fs_manager)
