"""
    This classifier class will perform basic experiments, receiving 
    as arguments the training arguments and the columns to use from the training dataset
"""
# Python imports
from typing             import Dict, List, Any, Tuple
from pathlib            import Path
from importlib          import resources
from enum               import Enum
# Third Party
from pandas.core.frame  import DataFrame
from transformers import (
    RobertaTokenizer,
    Trainer,
    TrainingArguments,
    RobertaForSequenceClassification,
)
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
from sklearn.model_selection import train_test_split
from datasets import Dataset
import torch
import pandas as pd
import numpy as np
import os

from transformers.trainer_utils import EvalPrediction

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

BASE_C4V_FOLDER = os.path.join(os.environ.get("HOME"), "/.c4v")
BASE_C4V_EXPERIMENTS_FOLDER = os.path.join(BASE_C4V_FOLDER, "/experiments")

class Tags(Enum):
    """
        Possible tags variants for an article
    """
    DENUNCIA_FALTA_DEL_SERVICIO = "DENUNCIA FALTA DEL SERVICIO"

class ClassifierExperiment:
    """
        This class provides a simple way to run simple experiments.
    """

    LOGS_FOLDER_NAME: str = "logs"
    RESULTS_EXPERIMENT_NAME: str = "results"
    RELEVANT_LABEL: str = "DENUNCIA FALTA DEL SERVICIO"

    def __init__(
        self,
        branch_name: str,
        experiment_name: str,
        test_dataset: str = "elpitazo_positivelabels_devdataset.csv",
        columns: List[str] = ["text"],
        base_path: str = BASE_C4V_EXPERIMENTS_FOLDER,
        use_cuda: bool = True,
        model_name: str = "mrm8488/RuPERTa-base",
        train_args: TrainingArguments = None,
    ):
        self._columns = columns
        self._branch_name = branch_name
        self._experiment_name = experiment_name
        self._experiments_folder = base_path
        self._device = (
            torch.device("cuda")
            if use_cuda and torch.cuda.is_available()
            else torch.device("cpu")
        )
        self._test_dataset = test_dataset
        self._model_name = model_name
        self._train_args = train_args

    def _get_files_path(self) -> str:
        """
            Get path to files for this experiment
        """
        return os.path.join(
            self._experiments_folder, f"{self._branch_name}/{self._experiment_name}"
        )

    def _get_path_to(self, folder: str) -> str:
        """
            Get path to given folder and create it if doesn't exist
        """
        if not os.path.exists(folder):
            Path(folder).mkdir(parents=True)

        return folder

    def get_logs_path(self) -> str:
        """
            Get path to logs for this experiment
            Return:
                path to log folder where we store logs for this experiment
        """
        return self._get_path_to(
            os.path.join(self._get_files_path(), f"{self.LOGS_FOLDER_NAME}")
        )

    def get_results_path(self) -> str:
        """
            Get path to results for this experiment
            Return:
                path to results folder where we store results for this experiment
        """
        return self._get_path_to(
            os.path.join(self._get_files_path(), f"{self.RESULTS_EXPERIMENT_NAME}")
        )

    def get_experiments_path(self) -> str:
        """
            Get path to experiments folder
        """
        return self._get_path_to(self._get_files_path())

    def get_dataframe(self, dataset_name: str = None) -> DataFrame:
        """
            Get dataframe as a pandas dataframe 
        """
        with resources.open_text(
            "data.raw.huggingface", dataset_name or self._test_dataset
        ) as f:
            return pd.read_csv(f)

    def prepare_dataframe(self) -> Tuple[List[str], List[int]]:
        """
            Return the list of text bodies and its corresponding label of whether it is 
            a missing service problem or not, expressed as int
            Return:
                ([str], [int]) = the i'st position of the first list is the body of a news article, and the 
                                 i'st position of the second list tells whether the article i talks about
                                 a missing service or not, expressed as an int (1 if it is, 0 if not)
        """

        df_elpitazo_pscdd = self.get_dataframe()
        df_elpitazo_pscdd["label"] = (
            df_elpitazo_pscdd.tipo_de_evento == Tags.DENUNCIA_FALTA_DEL_SERVICIO.value
        ).astype(int)

        df_elpitazo_pscdd = df_elpitazo_pscdd.convert_dtypes()
        df_issue_text = df_elpitazo_pscdd[[*self._columns, "label"]]
        df_issue_text.dropna(inplace=True)

        if len(self._columns) == 1:
            x = list(df_issue_text["text"])
        else:
            x = list(zip(*[list(df_issue_text[col]) for col in self._columns]))

        y = list(df_issue_text["label"])

        return x, y

    def load_model_from_hub(self) -> RobertaForSequenceClassification:
        """
            Return the model from hub, already configured
            Return:
                Model itself, already configured and loaded to device
        """
        model = RobertaForSequenceClassification.from_pretrained(self._model_name, num_labels=2)
        # send model to device
        model.to(self._device)

        return model

    def load_tokenizer_from_hub(self) -> RobertaTokenizer:
        """
            Load and configure tokenizer from hub
            Return:
                tokenizer from hub
        """
        tokenizer = RobertaTokenizer.from_pretrained(self._model_name)

        return tokenizer

    def transform_dataset(
        self, x: List[str], y: List[int], tokenizer: RobertaTokenizer
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
        X_train, X_val, y_train, y_val = train_test_split(x, y, test_size=0.2)

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

        model.save_pretrained(path_to_save_checkpoint or self._get_files_path())

        return trainer

    def load_fine_tuned_model(
        self, path: str = None
    ) -> RobertaForSequenceClassification:
        """
            load fine tuned model from provided path, defaults to an already trained one 
            in experiment's folder
        """
        path = path or os.path.join(self.get_experiments_path())
        model_path = os.path.join(path, "pytorch_model.bin")
        model = RobertaForSequenceClassification.from_pretrained(model_path)
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

    def run_experiment(self, train_args: Dict[str, Any] = None):
        """
            Run an experiment using the give training args
        """
        # Prepare dataframe and load model + tokenizer
        x, y = self.prepare_dataframe()

        model = self.load_model_from_hub()
        tokenizer = self.load_tokenizer_from_hub()

        train_dataset, val_dataset = self.transform_dataset(x, y, tokenizer)

        # Fine tune the model
        fine_tuned_model_trainer = self.train_and_save_model(
            model=model,
            output_dir=self.get_results_path(),
            logging_dir=self.get_logs_path(),
            path_to_save_checkpoint=self.get_experiments_path(),
            train_args=self._override_train_args(train_args or {}),
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
        )

        # Get the metrics from the model
        metrics_df = self.evaluate_metrics(
            trainer=fine_tuned_model_trainer, val_dataset=val_dataset
        )

        return metrics_df

