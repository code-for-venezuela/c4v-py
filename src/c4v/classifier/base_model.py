"""
    This class represents a model, you can extend it to have some common operations 
    for models, such as managing a files folder, or loading datasets from the data
    folder
"""
# Python imports
from pathlib import Path
from importlib import resources
import pandas as pd

# Third party imports
import torch
from transformers import PreTrainedTokenizer, AutoTokenizer, PreTrainedModel
from transformers.utils.dummy_pt_objects import AutoModel

# Local imports
from c4v.config import settings


class BaseModel:
    """
        Represents a model and contains useful utilities for models, such
        as managing files folders and loading datasets from the data folder.
    """

    RESULTS_EXPERIMENT_NAME = "results"
    LOGS_FOLDER_NAME = "logs"

    def __init__(
        self,
        files_folder_path: str = None,
        base_model_name: str = settings.default_base_language_model,
        use_cuda: bool = True,
    ) -> None:

        # Set up cuda
        self._device = (
            torch.device("cuda")
            if use_cuda and torch.cuda.is_available()
            else torch.device("cpu")
        )

        # Set up base model and files folder path
        self._base_model_name = base_model_name
        self.files_folder_path = files_folder_path

        # Init lazy tokenizer and model variables:
        self._model = None
        self._tokenizer = None

    @property
    def device(self):
        """
            Device used when evaluating the model
        """
        self._device

    @property
    def files_folder_path(self) -> str:
        """
            Folder to store files for a training process
        """
        return self._files_folder

    @files_folder_path.setter
    def files_folder_path(self, value: str):
        # Check that folder for internal files does exists
        if value and not Path(value).exists():
            raise ValueError(f"Given path does not exists: {value}")
        self._files_folder = value

    @property
    def logs_path(self) -> str:
        """
            Get path to logs for this experiment
            Return:
                path to log folder where we store logs for this experiment
        """
        if not self.files_folder_path:
            raise ValueError(
                "Could not create logs files, as no files folder is configured for this classifier object"
            )

        p = Path(self.files_folder_path, f"{self.LOGS_FOLDER_NAME}")

        # Create folder if does not exists
        if not p.exists():
            try:
                p.mkdir()
            except IOError as e:
                raise ValueError(
                    f"Could not create logs folder for path: {str(p)}, error: {e}"
                )

        return str(p)

    @property
    def results_path(self) -> str:
        """
            Get path to results for this experiment
            Return:
                path to results folder where we store results for this experiment
        """
        if not self.files_folder_path:
            raise ValueError(
                "Could not create results files, as no files folder is configured for this classifier object"
            )

        p = Path(self._files_folder, f"{self.RESULTS_EXPERIMENT_NAME}")

        # Create folder if does not exists
        if not p.exists():
            try:
                p.mkdir()
            except IOError as e:
                raise ValueError(
                    f"Could not create logs folder for path: {str(p)}, error: {e}"
                )

        return str(p)

    @property
    def tokenizer(self) -> PreTrainedTokenizer:
        """
            Internal tokenizer object. It's lazy-loaded, so it will
            be loaded once when it's called for the first time
        """
        if self._tokenizer == None:
            self._tokenizer = AutoTokenizer.from_pretrained(self._base_model_name)

        return self._tokenizer

    @property
    def model(self) -> PreTrainedModel:
        """
            Internal model object. It's lazy-loaded, so it will be loaded once when it's called for the first time
        """
        if self._model == None:
            self._model = AutoModel.from_pretrained(self._base_model_name)

        return self._model


class C4vDataFrameLoader:
    """
        Use this object to get data from the data folder as pandas dataframes
    """

    @staticmethod
    def get_from_raw(dataset_name: str) -> pd.DataFrame:
        """
            Get dataframe as a pandas dataframe, using a csv file stored in <project_root>/data/raw/huggingface
        """
        with resources.open_text("data.raw.huggingface", dataset_name) as f:
            return pd.read_csv(f)

    @staticmethod
    def get_from_processed(dataset_name: str) -> pd.DataFrame:
        """
            Get dataframe as a pandas dataframe, using a csv file stored in <project_root>/data/processed/huggingface
        """
        with resources.open_text("data.processed.huggingface", dataset_name) as f:
            return pd.read_csv(f)
