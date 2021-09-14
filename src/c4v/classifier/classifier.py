"""
    This classifier class will perform basic experiments, receiving 
    as arguments the training arguments and the columns to use from the training dataset
"""
# Local imports
from c4v.config import settings
from c4v.scraper.scraped_data_classes.scraped_data import ScrapedData

# Python imports
from typing import Dict, List, Any, Tuple
from pathlib import Path
from pandas.core.frame import DataFrame
from importlib import resources
from enum import Enum

# Third Party
from pandas.core.frame  import DataFrame
from transformers import (
    RobertaTokenizer,
    Trainer,
    TrainingArguments,
    RobertaForSequenceClassification,
    AutoModelForSequenceClassification,
    AutoTokenizer, 
    AutoModelForMaskedLM
)
from transformers_interpret import SequenceClassificationExplainer
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
from sklearn.model_selection import train_test_split
from datasets import Dataset
import torch
import pandas as pd
import numpy as np

from transformers.trainer_utils import EvalPrediction

BASE_C4V_FOLDER = settings.c4v_folder

class Labels(Enum):
    """
        Every possible label for each article
    """

    DENUNCIA_FALTA_DEL_SERVICIO = "DENUNCIA FALTA DEL SERVICIO"
    IRRELEVANTE = "IRRELEVANTE"

    @classmethod
    def labels(cls) -> List[str]:
        """
            Get list of labels as strings
        """
        return [l.value for l in cls]

class Tags(Enum):
    """
        Possible tags variants for an article
    """
    DENUNCIA_FALTA_DEL_SERVICIO = "DENUNCIA FALTA DEL SERVICIO"

class Classifier:
    """
        This class provides a simple way to run simple experiments.
    """

    LOGS_FOLDER_NAME = "logs"
    RESULTS_EXPERIMENT_NAME = "results"

    def __init__(
        self,
        use_cuda: bool = True,
        base_model_name: str = "BSC-TeMU/roberta-base-bne",
        files_folder: str = None,
    ):
        self._device = (
            torch.device("cuda")
            if use_cuda and torch.cuda.is_available()
            else torch.device("cpu")
        )
        self._base_model_name = base_model_name
        self.files_folder = files_folder

    @property
    def files_folder(self) -> str:
        """
            Folder to store files for a training process
        """
        return self._files_folder

    @files_folder.setter
    def files_folder(self, value: str):
        # Check that folder for internal files does exists
        if value and not Path(value).exists():
            raise ValueError(f"Given path does not exists: {value}")
        self._files_folder = value

    def get_logs_path(self) -> str:
        """
            Get path to logs for this experiment
            Return:
                path to log folder where we store logs for this experiment
        """
        if not self._files_folder:
            raise ValueError(
                "Could not create logs files, as no files folder is configured for this classifier object"
            )

        p = Path(self._files_folder, f"{self.LOGS_FOLDER_NAME}")

        # Create folder if does not exists
        if not p.exists():
            try:
                p.mkdir()
            except IOError as e:
                raise ValueError(
                    f"Could not create logs folder for path: {str(p)}, error: {e}"
                )

        return str(p)

    def get_results_path(self) -> str:
        """
            Get path to results for this experiment
            Return:
                path to results folder where we store results for this experiment
        """
        if not self._files_folder:
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

    def get_dataframe(self, dataset_name: str) -> DataFrame:
        """
            Get dataframe as a pandas dataframe, using a csv file stored in <project_root>/data/raw/huggingface
        """
        with resources.open_text("data.processed.huggingface", dataset_name) as f:
            return pd.read_csv(f)

    def prepare_dataframe(
        self, columns: List[str], dataset_name: str
    ) -> Tuple[List[str], List[int]]:
        """
            Return the list of text bodies and its corresponding label of whether it is 
            a missing service problem or not, expressed as int
            Parameters:
                columns : [str] = List of columns to use as part of the experiment 
                dataset_name : str = name of the dataset to use
            Return:
                ([str], [int]) = the i'st position of the first list is the body of a news article, and the 
                                 i'st position of the second list tells whether the article i talks about
                                 a missing service or not, expressed as an int (1 if it is, 0 if not)
        """

        df_pscdd = self.get_dataframe(dataset_name).sample(frac=1) # use sample to shuffle rows

        df_pscdd["label"] = (
            df_pscdd['label'].apply(lambda x: 'IRRELEVANTE' not in x)
        ).astype(int)

        df_pscdd = df_pscdd.convert_dtypes()
        df_issue_text = df_pscdd[[*columns, "label"]]
        df_issue_text.dropna(inplace=True)

        x = [
            "\n".join(tup)
            for tup in zip(*[list(df_issue_text[col]) for col in columns])
        ]

        y = list(df_issue_text["label"])

        return x, y

    def load_tokenizer_from_hub(self) -> RobertaTokenizer:
        """
            Create & configure tokenizer from hub
            Return:
                RobertaTokenizer: tokenizer to retrieve
        """
        return AutoTokenizer.from_pretrained(
            self._base_model_name, id2label=self.get_id2label_dict()
        )

    def load_base_model_from_hub(
        self,
    ) -> Tuple[RobertaForSequenceClassification, RobertaTokenizer]:
        """
            Create model from model hub, configure them and retrieve it
            Return:
                RobertaForSequenceClassification : the model as specified
        """
        # Creating model and tokenizer
        model = AutoModelForSequenceClassification.from_pretrained(
            self._base_model_name, num_labels=2
        )
        # Use GPU if available
        model.to(self._device)

        return model

    def transform_dataset(
        self,
        x: List[str],
        y: List[int],
        tokenizer: RobertaTokenizer,
        test_dataset_proportion: float = 0.2,
    ) -> Tuple[Dataset, Dataset]:
        """
            perform operations needed to post process a dataset separated in input list
            "x" and expected answers list "y", returning a training and a validation datasets,
            ready to be fed into the model 
            Parameters:
                x : [str] = List of content inputs to train with
                y : [int] = List of integer values matching answer for inputs in x list
            Return:
                (Dataset, Dataset) = the training and the validation dataset, in such order
        """
        # Train Test Split
        X_train, X_val, y_train, y_val = train_test_split(
            x, y, test_size=test_dataset_proportion
        )

        X_train_tokenized = tokenizer(
            X_train, padding=True, truncation=True, max_length=512
        )

        X_val_tokenized = tokenizer(
            X_val, padding=True, truncation=True, max_length=512
        )

        # Create torch dataset
        class _Dataset(torch.utils.data.Dataset):
            def __init__(self, encodings, labels=None):
                self.encodings = encodings
                self.labels = labels

            def __getitem__(self, idx):
                item = {
                    key: torch.tensor(val[idx]) for key, val in self.encodings.items()
                }
                if self.labels:
                    item["labels"] = torch.tensor(self.labels[idx])
                return item

            def __len__(self):
                return len(self.encodings["input_ids"])

        train_dataset = _Dataset(X_train_tokenized, y_train)
        val_dataset = _Dataset(X_val_tokenized, y_val)

        return train_dataset, val_dataset

    @staticmethod
    def compute_metrics(predictions: EvalPrediction) -> Dict[str, Any]:
        """
            Use this function to evaluate metrics for a given prediction 
            Parameters:
                predictions : EvalPrediction = Prediction provided by the model with certain input 
            Return:
                Dict with computed metrics. Provided fields:
                    - accuracy
                    - precision
                    - recall
                    - f1
        """
        pred, labels = predictions
        pred = np.argmax(pred, axis=1)

        accuracy = accuracy_score(y_true=labels, y_pred=pred)
        recall = recall_score(y_true=labels, y_pred=pred)
        precision = precision_score(y_true=labels, y_pred=pred)
        f1 = f1_score(y_true=labels, y_pred=pred)

        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }

    def _get_default_train_args(self) -> Dict[str, Any]:
        """
            Return a default version of training arguments as a dict
        """

        default = {
            "output_dir": self.get_results_path(),
            "num_train_epochs": 1,
            "per_device_train_batch_size": 10,
            "per_device_eval_batch_size": 10,
            "warmup_steps": 500,
            "weight_decay": 0.01,
            "logging_dir": self.get_logs_path(),
            "save_total_limit": 1,
        }

        return default

    def _override_train_args(self, new_args: Dict[str, Any] = {}) -> Dict[str, Any]:
        """
            Return the default args dict, with overriden settings specified as the ones specified in 
            input dict
        """

        default = self._get_default_train_args()
        for (k, v) in new_args.items():
            default[k] = v

        return default

    def train_and_save_model(
        self,
        model: Any,
        output_dir: str = None,
        logging_dir: str = None,
        path_to_save_checkpoint: str = None,
        train_args: Dict[str, Any] = None,
        train_dataset: Dataset = None,
        eval_dataset: Dataset = None,
    ) -> Trainer:
        """
            Train and save using provided model, using provided args and storing 
            results to "output_dir" and logging additional info to "logging_dir",
            saving checkpoint to "path_to_save_checkpoint". If some of this values is not provided,
            it will be defaulted to this experiment's dedicated folder 
            Parameters:
                model = model to train
                output_dir : str = results folder, defaulted to experiment's results folder
                logging_dir : str = logs folder, defaulted to experiment's logs folder
                path_to_save_checkpoint : str = path to save checkpoint after training is finished, defaults to experiment's folder
                train_args : TrainArguments = if provided, override all default parameters passed to the training
            Return:
                properly configured trainer instance

        """
        if output_dir:
            train_args["output_dir"] = output_dir
        if logging_dir:
            train_args["logging_dir"] = logging_dir

        args = TrainingArguments(**self._override_train_args(train_args))

        trainer = Trainer(
            model=model,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            compute_metrics=self.__class__.compute_metrics,
        )

        # Train pre-trained model
        trainer.train()

        model.save_pretrained(path_to_save_checkpoint or self.files_folder)

        return trainer

    def load_fine_tuned_model(
        self, path: str = None
    ) -> RobertaForSequenceClassification:
        """
            load fine tuned model from provided path, defaults to an already trained one 
            in experiment's folder
        """
        path = path or self._files_folder

        # Check if path is a valid one
        if not Path(path, "config.json").exists():
            raise ValueError(f"Experiment does not exists: {path}")

        model = AutoModelForSequenceClassification.from_pretrained(
            path, local_files_only=True, id2label=self.get_id2label_dict()
        )
        return model

    def evaluate_metrics(self, trainer: Trainer, val_dataset: Dataset) -> DataFrame:
        """
            Compute metrics as a dataframe for this trainer object using the validation dataset
            Parameters:
                trainer : Trainer = trainer object already trained and ready to test
                val_dataset : Dataset = data used for validation
            Return:
                A dataframe with output of a validation
        """
        ## Evaluate Metrics
        metrics = trainer.evaluate(val_dataset)
        metrics_df = pd.DataFrame.from_dict(
            metrics, orient="index", columns=["metrics_value"]
        )
        return metrics_df

    def run_training(
        self,
        train_args: Dict[str, Any] = None,
        columns: List[str] = ["text"],
        dataset: str = "elpitazo_positivelabels_devdataset.csv",
    ) -> Dict[str, Any]:
        """
            Run an experiment specified by given train_args, and write a summary if requested so
            Parameters:
                train_args : Dict[str, Any] = arguments passed to trainig arguments
                columns : [str] = columns to use in the dataset
                dataset : dataset tu use during training, should be a name of a dataset under <project_root>/data/raw/huggingface
            Return:
                Classifier metrics
        """

        if not self._files_folder:
            raise ValueError(
                "Can't train in a Classifier without a folder for local data"
            )

        # Prepare dataframe and load model + tokenizer
        x, y = self.prepare_dataframe(columns=columns, dataset_name=dataset)

        model = self.load_base_model_from_hub()
        tokenizer = self.load_tokenizer_from_hub()

        train_dataset, val_dataset = self.transform_dataset(x, y, tokenizer)

        # Fine tune the model
        fine_tuned_model_trainer = self.train_and_save_model(
            model=model,
            output_dir=self.get_results_path(),
            logging_dir=self.get_logs_path(),
            path_to_save_checkpoint=self._files_folder,
            train_args=self._override_train_args(train_args or {}),
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
        )

        # Get the metrics from the model
        metrics_df = self.evaluate_metrics(
            trainer=fine_tuned_model_trainer, val_dataset=val_dataset
        )

        return metrics_df

    def classify(
        self, data: ScrapedData, model: str = None
    ) -> Dict[str, Any]:  # @TODO should use a bulk version instead
        """
            Classify the given data instance, returning classification metrics
            as a simple dict.
            Parameters:
                data : ScrapedData = Data instance to classify
                model : str = model name of model to load use when classifying. If no model provided,
                              use the model configured for this classifier
            Return:
                Dict with classification data, predicted label and score for each possible label
        """
        # Get model from experiment:
        if model is None:
            model = self._files_folder
            # Check if there's a config.json in this folder
            if not Path(model, "config.json").exists():
                raise ValueError(f"path: {model} does not contains valid model to load")

        roberta_model = self.load_fine_tuned_model(model)

        # Tokenize input
        roberta_tokenizer = self.load_tokenizer_from_hub()
        tokenized_input = roberta_tokenizer(
            [data.title],
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )

        raw_output = roberta_model(**tokenized_input)
        output = torch.nn.functional.softmax(raw_output.logits, dim=-1)

        label_id = torch.argmax(output).item()
        return {"label": self.index_to_label(label_id), "scores": output.tolist()}

    def explain(
        self, sentence: str, html_file: str = None, additional_label: str = None
    ) -> Dict[str, Any]:
        """
            Return a list of words from provided sentence with how much they collaborate to each label 
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
        # Load model and tokenizer
        model = self.load_fine_tuned_model()
        tokenizer = self.load_tokenizer_from_hub()

        # Create explainer
        explainer = SequenceClassificationExplainer(model, tokenizer)

        scores = explainer(sentence, class_name=additional_label)
        label = explainer.predicted_class_name

        # write html file
        if html_file:
            explainer.visualize(html_file)

        return {"scores": scores, "label": label}

    def get_id2label_dict(self) -> Dict[int, str]:
        """
            Return dict mapping from ids to labels
        """
        return {
            1: Labels.DENUNCIA_FALTA_DEL_SERVICIO.value,
            0: Labels.IRRELEVANTE.value,
        }

    def index_to_label(self, index: int) -> Labels:
        """
            Get index for label
        """
        d = self.get_id2label_dict()

        return d.get(index, Labels.IRRELEVANTE.value)

    @staticmethod
    def get_labels() -> List[str]:
        """
            Get list of possible labels outputs
        """
        return Labels.labels()
