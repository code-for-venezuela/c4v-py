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
from c4v.scraper.scraped_data_classes.scraped_data import ScrapedData

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

    def __init__(self, experiment_fs_manager : experiment.ExperimentFSManager, classifier_instance : classifier.Classifier = None):
        # Initialize as super class
        super().__init__(experiment_fs_manager=experiment_fs_manager)

        # Set up default values
        if not classifier_instance:
            classifier_instance = classifier.Classifier()

        # Set up classifier object
        classifier_instance.files_folder = experiment_fs_manager.experiment_content_folder
        self._classifier = classifier_instance

    def experiment_to_run(self, args: ClassifierArgs) -> ClassifierSummary:
        # Run a training process
        metrics = self._classifier.run_train(args.training_args)
        summary = ClassifierSummary(eval_metrics=metrics, description=args.description)

        return summary

    def classify(self, data : ScrapedData) -> Dict[str, Any]:
        """
            Classify this sentence using configured experiment
            Parameters:
                sentence : str = sentence or text to be classifier
            Return:
                Predicted label and score for every other label
        """
        return self._classifier.classify(data)

    def explain(self, sentence : str, html_file : str = None, additional_label : str = None) -> Dict[str, Any]:
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
