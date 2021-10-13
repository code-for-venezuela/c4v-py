"""
    This experiment is intended to train a classification 
    model with local data.
"""
# Python imports
import dataclasses
from typing import Dict, Any, List

# Local imports
from c4v.classifier.experiment import (
    ExperimentFSManager,
    BaseExperimentSummary,
    BaseExperimentArguments,
    BaseExperiment,
)
from c4v.classifier.classifier import Classifier
from c4v.scraper.scraped_data_classes.scraped_data import ScrapedData
from c4v.config import settings


@dataclasses.dataclass
class ClassifierArgs(BaseExperimentArguments):
    """
        Arguments passed to the model during training
    """

    training_args: Dict[str, Any] = dataclasses.field(
        default_factory=dict
    )  # training arguments passed to the trained object during training
    columns: List[str] = dataclasses.field(
        default_factory=lambda: ["text"]
    )  # fields from ScrapedData to use
    train_dataset_name: str = "classifier_training_dataset.csv"  # name of the training dataset stored in data.processed.huggingface
    confirmation_dataset_name: str = "classifier_confirmation_dataset.csv"  # name of the confirmation dataset stored in data.processed.huggingface
    description: str = None  # Optional description for this experiment
    val_dataset_proportion: float = 0.2  # How much in proportion for the training dataset take as eval dataset
    base_model_name: str = settings.default_base_language_model


@dataclasses.dataclass
class ClassifierSummary(BaseExperimentSummary):
    """
        Metrics returned by the classifier after every run 
    """

    eval_metrics: Dict[str, Any] = dataclasses.field(default_factory=dict)
    user_args: ClassifierArgs = ClassifierArgs()

    def __str__(self) -> str:
        super_str = super().__str__()
        # Add eval_metrics fields
        super_str += "EVAL METRICS:\n"
        super_str += (
            "\n".join(
                [
                    f"\t* {k} = {v}"
                    for (k, v) in self.eval_metrics["metrics_value"].items()
                ]
            )
            + "\n"
        )
        super_str += "USER ARGUMENTS:\n"
        super_str += "\tColumns: \n"
        super_str += (
            "\n".join([f"\t\t* {c}" for c in self.user_args.columns])
            if self.user_args.columns
            else "\t\t<No Columns Provided>"
        )
        super_str += "\n"
        super_str += f"\tTrain Dataset: {self.user_args.train_dataset_name}\n"
        super_str += (
            f"\tConfirmation Dataset: {self.user_args.confirmation_dataset_name}\n"
        )
        super_str += f"\tValidation Dataset Proportion: {self.user_args.val_dataset_proportion}\n"
        super_str += f"\tBase Model Name: {self.user_args.base_model_name}\n"
        super_str += "\tTraining Arguments:\n"
        super_str += (
            "\n".join(
                [f"\t\t* {k} = {v}" for (k, v) in self.user_args.training_args.items()]
            )
            + "\n"
        )

        return super_str


class ClassifierExperiment(BaseExperiment):
    """
        Use this experiment to run a classifier training. 
        You can check valid training arguments here:
        https://huggingface.co/transformers/main_classes/trainer.html#transformers.TrainingArguments
    """

    def __init__(
        self,
        experiment_fs_manager: ExperimentFSManager,
        classifier_instance: Classifier = None,
    ):
        # Initialize super class
        super().__init__(experiment_fs_manager=experiment_fs_manager)

        # Set up default values
        if not classifier_instance:
            classifier_instance = Classifier()

        # Set up classifier object
        classifier_instance.files_folder = (
            experiment_fs_manager.experiment_content_folder
        )

        self._classifier = classifier_instance

    @property
    def classifier(self) -> Classifier:
        """
            Configured classifier instance
        """
        return self._classifier

    def experiment_to_run(self, args: ClassifierArgs) -> ClassifierSummary:
        # Run a training process
        metrics = self._classifier.run_training(
            args.training_args,
            columns=args.columns,
            training_dataset=args.train_dataset_name,
            confirmation_dataset=args.confirmation_dataset_name,
            val_test_proportion=args.val_dataset_proportion,
            base_model_name=args.base_model_name,
        )
        summary = ClassifierSummary(
            eval_metrics=metrics, description=args.description, user_args=args
        )
        print(summary)
        return summary

    def classify(self, data: List[ScrapedData]) -> List[Dict[str, Any]]:
        """
            Classify this sentence using configured experiment
            Parameters:
                sentence : str = sentence or text to be classifier
            Return:
                A List of dicts with the resulting scraped data correctly labelled
                and its corresponding scores tensor for each possible label. Available fields:
                    + data : ScrapedData = resulting data instance after classification
                    + scores : torch.Tensor = Scores for each label returned by the classifier
        """
        return self._classifier.classify(data)

    def explain(
        self, sentence: str, html_file: str = None, additional_label: str = None
    ) -> Dict[str, Any]:
        """
            Explain given sentence using provided model
            Parameters:
                sentence : str = text to explain
                html_file : str = path to some html file to store human readable representation. If no provided, 
                                    it's ignored
                additional_label : str = Label to include in expalantion. If the predicted label is different 
                                         from this one, then explain how much this label was contributing to 
                                         its corresponding value. Ignored if not provided.
            Return:
                Dict with data for this explanation. For example:
                {   "scores" : 
                    [   
                        ('denuncian' , 0.98932),
                        ('falta'     , 0.78912),
                        ('de'        , 0.001231),
                        ('agua'      , 0.863781)
                    ],
                    "label" : "DENUNCIA_FALTA_DEL_SERVICIO",
        """
        return self._classifier.explain(sentence, html_file, additional_label)

    @classmethod
    def from_branch_and_experiment(
        cls,
        branch_name: str,
        experiment_name: str,
        classifier_instance: Classifier = None,
    ):
        """
            Create an experiment from a branch name, an experiment name, and a classifier instance
            Parameters:
                branch_name : str = Branch name for the experiment
                experiment_name : str = Experiment name
                classifier_instance : Classifier = optional classifier instance, will be defaulted if none was provided
        """
        fs_manager = ExperimentFSManager(branch_name, experiment_name)

        if classifier_instance:
            classifier_instance.files_folder_path = fs_manager.experiment_content_folder

        classifier_instance = classifier_instance or Classifier(
            fs_manager.experiment_content_folder
        )
        return cls(fs_manager, classifier_instance)
