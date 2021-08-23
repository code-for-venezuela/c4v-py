"""
    An experiment represents a process ran by demand by a developer,
    which may need some special files and folders, and access to other
    components of the c4v library, the PersistencyManager class for example. 
    Experiments are required to provide a branch name and an experiment name, 
    so they can be organized and searched. Also, experiments are provided with 
    a folder for their necessary files.

    Also, experiments should provide a summary for results, and an Args Object to use 
    when they run
"""
# Python imports
import dataclasses
import pytz
from pathlib     import Path
from datetime    import datetime

# Local imports
from c4v.config import settings

dataclasses.dataclass(frozen=True)
class BaseExperimentSummary:
    """
        Provide an output summary describing results for this experiment, 
        and an human readable representation
    """
    # Initialize to date when created
    date : datetime = dataclasses.field(default_factory=lambda: datetime.now(tz=pytz.utc))

    # Optional description
    description : str = None
    
    def __str__(self) -> str:
        # An human readable representation 
        output =  f"Date: {datetime.strftime( self.date, settings.date_format )}"
        output += f"Description: {self.description if self.description else '<no description>'}"

        return output


@dataclasses.dataclass
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
    def __init__(self, base_folder : str) -> None:
        assert Path(base_folder).exists(), f"Folder {base_folder} does not exists" # Checks if path exists
        self._base_folder = base_folder

    def run_experiment(self, args : BaseExperimentArguments) -> BaseExperimentSummary:
        """
            Run an experiment and return a summary for this experiment
        """
    raise NotImplementedError("Should implement run_experiment abstract class")
    