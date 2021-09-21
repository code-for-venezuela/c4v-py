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
from c4v.classifier.experiment import BaseExperiment, BaseExperimentSummary, BaseExperimentArguments, ExperimentFSManager
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
    model_name      : str = None
    use_cuda        : bool = True
    train_dataset   : Dataset = None
    eval_dataset    : Dataset  = None
    train_args      : Dict[str, Any] = dataclasses.field(default_factory=dict)

class LanguageModelExperimentSummary(BaseExperimentSummary):
    """
        Language model experiment summary
    """
    user_args : LanguageModelExperimentArguments = LanguageModelExperimentArguments()
    result : pd.DataFrame = None


class LanguageModelExperiment(BaseExperiment):
    """
        Experiment to run a training for the language model
    """
    def experiment_to_run(self, args: LanguageModelExperimentArguments) -> LanguageModelExperimentSummary:
        lang_model = LanguageModel(
                files_folder_path=self._experiment_fs_manager.experiment_content_folder, 
                base_model_name=args.model_name, 
                use_cuda=args.use_cuda
            )

        metrics_df = lang_model.run_training(
            train_dataset=args.train_dataset,
            eval_dataset=args.eval_dataset,
            train_args=args.train_args,
        )

        return metrics_df

    