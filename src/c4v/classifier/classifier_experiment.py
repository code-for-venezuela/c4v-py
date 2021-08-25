"""
    This experiment is intended to train a classification 
    model with local data.
"""
# Python imports
import dataclasses
from typing import Dict, Any

# Local imports
import c4v.classifier.experiment as experiment
import c4v.classifier.classifier as classifier


@dataclasses.dataclass
class ClassifierSummary(experiment.BaseExperimentSummary):
    """
        Metrics returned by the classifier after every run 
    """
    eval_metrics: Dict[str, Any] = dataclasses.field(default_factory=dict)

    def __str__(self) -> str:
        super_str = super().__str__()    
        # Add eval_metrics fields
        super_str += "EVAL METRICS:\n"
        super_str += "\n".join([f"\t* {k} = {v}" for (k,v) in self.eval_metrics['metrics_value'].items()]) + "\n"

        return super_str

@dataclasses.dataclass
class ClassifierArgs(experiment.BaseExperimentArguments):
    """
        Arguments passed to the model during training
    """
    training_args : Dict[str, Any] = dataclasses.field(default_factory=dict)
    description   : str = None

class ClassifierExperiment(experiment.BaseExperiment):
    """
        Use this experiment to run a classifier training
    """

    def __init__(self, base_folder: str = None, classifier_args : Dict[str, Any] = {}):
        # Initialize as super class
        super().__init__(base_folder=base_folder)

        # Set up classifier object
        classifier_args['files_folder'] = base_folder
        self._classifier = classifier.Classifier(**classifier_args)

    @property
    def base_folder(self) -> str:
        return experiment.BaseExperiment.base_folder.fget()

    @base_folder.setter
    def base_folder(self, new_val : str):
        experiment.BaseExperiment.base_folder.fset(self, new_val)
        self._classifier.files_folder = new_val

    def run_experiment(self, args: ClassifierArgs) -> ClassifierSummary:
        # Run a training process
        metrics = self._classifier.run_train(args.training_args)
        summary = ClassifierSummary(eval_metrics=metrics, description=args.description)

        return summary

