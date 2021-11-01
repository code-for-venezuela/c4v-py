"""
    This experiment will run a base language model training
"""
# Python imports
import dataclasses
from typing import Any, Dict, List

# Third party imports
from torch.utils.data import Dataset
import pandas as pd

# Local imports
from c4v.classifier.experiment import (
    BaseExperiment,
    BaseExperimentSummary,
    BaseExperimentArguments,
    ExperimentFSManager,
)
from c4v.classifier.language_model.language_model import LanguageModel


@dataclasses.dataclass
class LanguageModelExperimentArguments(BaseExperimentArguments):
    """
        Arguments for Language model experiment
        You can check valid training arguments here:
        https://huggingface.co/transformers/main_classes/trainer.html#transformers.TrainingArguments
        Note:
            You can easlily build a dataset using the LanguageModel.create_dataset_from_scraped_data function 
                 
    """

    model_name: str = None  # Base model name to load
    use_cuda: bool = True  # If should use cuda when available. Ignored when not available
    train_dataset: Dataset = None  # Training dataset
    eval_dataset: Dataset = None  # Evaluation dataset (used during training)
    confirmation_dataset: Dataset = None  # Confirmation dataset (not used during training, used afterwards to validate training)
    train_args: Dict[str, Any] = dataclasses.field(
        default_factory=dict
    )  # Training arguments passed to the Trainer object
    description: str = None  # Optional description


@dataclasses.dataclass
class LanguageModelExperimentSummary(BaseExperimentSummary):
    """
        Language model experiment summary
    """

    user_args: LanguageModelExperimentArguments = LanguageModelExperimentArguments()
    result: pd.DataFrame = None

    def __str__(self) -> str:
        s = super().__str__().rstrip() + "\n"

        # User arguments
        s += "USER ARGS:\n"
        s += f"\tmodel_name : {self.user_args.model_name}\n"
        s += f"\tuse_cuda : {self.user_args.use_cuda}\n"
        s += "\ttrain_args : {\n"
        for (k, v) in self.user_args.train_args.items():
            s += f"\t\t{k} : {v}\n"
        s += "\t}\n"

        # Result:
        s += "RESULTS:\n"
        s += str(self.result)

        return s


class LanguageModelExperiment(BaseExperiment):
    """
        Experiment to run a training for the language model
    """

    def experiment_to_run(
        self, args: LanguageModelExperimentArguments
    ) -> LanguageModelExperimentSummary:
        lang_model = LanguageModel(
            files_folder_path=self._experiment_fs_manager.experiment_content_folder,
            base_model_name=args.model_name,
            use_cuda=args.use_cuda,
        )

        metrics_df = lang_model.run_training(
            train_dataset=args.train_dataset,
            eval_dataset=args.eval_dataset,
            train_args=args.train_args,
            confirmation_dataset=args.confirmation_dataset,
        )
        summary = LanguageModelExperimentSummary(
            description=args.description, result=metrics_df, user_args=args
        )
        return summary
