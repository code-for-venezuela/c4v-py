"""
    This classifier class will perform basic experiments, receiving 
    as arguments the training arguments and the columns to use from the training dataset
"""
# Local imports
import dataclasses
from c4v.config import settings
from c4v.scraper.scraped_data_classes.scraped_data import ScrapedData, LabelSet, RelevanceClassificationLabels
from c4v.classifier.base_model import BaseModel, C4vDataFrameLoader

# Python imports
from typing import Dict, List, Any, Tuple, Type
from pathlib import Path
from pandas.core.frame import DataFrame
from enum import Enum

# Third Party
from pandas.core.frame import DataFrame
from transformers import (
    RobertaTokenizer,
    Trainer,
    TrainingArguments,
    RobertaForSequenceClassification,
    AutoModelForSequenceClassification,
    AutoTokenizer,
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
BASE_LANGUAGE_MODEL = settings.default_base_language_model

class LabelSet(Enum):
    """
        Interface for sets of labels that can be attached to to a model for classification
    """

    @classmethod
    def get_id2label_dict(cls) -> Dict[int, str]:
        """
            Get a dict mapping from ids to a str, representing the labels for this label set
        """
        raise NotImplementedError("Should implement abstract method get_id2label_dict")

class BinaryClassificationLabels(LabelSet):
    """
        Labels for Binary classification, telling if a data instance is relevant or not
    """
    IRRELEVANTE: str = "IRRELEVANTE"
    DENUNCIA_FALTA_DEL_SERVICIO: str = "PROBLEMA DEL SERVICIO"
    UNKNOWN: str = "UNKNOWN"


    @classmethod
    def get_id2label_dict(cls) -> Dict[int, str]:
        return {
            0: cls.IRRELEVANTE.value,
            1: cls.DENUNCIA_FALTA_DEL_SERVICIO.value,
        }


class Classifier(BaseModel):
    """
        This is the classifier model, you can use it to do two kinds of classification,
        binary classification, to tell apart relevant or irrelevant news, and a multi-single label classification,
        which assumes that an article is relevant and assigns one of a given set of labels to that
    """

    def __init__(
            self, 
            files_folder_path: str = None, 
            base_model_name: str = settings.default_base_language_model, 
            use_cuda: bool = True, 
            labelset : Type[LabelSet] = RelevanceClassificationLabels,
            label_column : str = "label"
            ):
        self._label_column = label_column
        self._labelset = labelset
        super().__init__(files_folder_path=files_folder_path, base_model_name=base_model_name, use_cuda=use_cuda)

    @property
    def label_column(self) -> str:
        """ Column used by this model as target label during training """
        return self._label_column

    @property
    def labelset(self) -> Type[LabelSet]:
        return self._labelset

    def get_dataframe(self, dataset_name: str) -> DataFrame:
        """
            Get dataframe as a pandas dataframe, using a csv file stored in <project_root>/data/processed/huggingface
        """
        return C4vDataFrameLoader.get_from_processed(dataset_name)

    def prepare_dataframe(
        self, columns: List[str], dataset_name: str, label_column: str = None, labelset: Type[LabelSet] = None
    ) -> Tuple[List[str], List[int]]:
        """
            Return the list of text bodies and its corresponding label corresponding according to the provided 
            labelset. Note that no duplicates are allowed
            Parameters:
                columns : [str] = List of columns to use as part of the experiment 
                dataset_name : str = name of the dataset to use
                label_column : str = name of the column to use as label. If not provided, defaults to the configured one
                                     in this classifier instance
                labelset : Type[LabelSet] = Set of label to use for training, there's no check that the provided label column 
                                            values  matches the labels in this labelset. If not provided, defaults to the configured one
                                            in this classifier instance
            Return:
                ([str], [int]) = the i'st position of the first list is the body of a news article, and the 
                                 i'st position of the second list tells whether the article i talks about
                                 a missing service or not, expressed as an int (1 if it is, 0 if not)
        """

        label_column = label_column or self.label_column
        labelset = labelset or self.labelset

        df = self.get_dataframe(dataset_name).sample(
            frac=1
        )  # use sample to shuffle rows

        label_2_id_dict = labelset.get_label2id_dict()

        df[label_column] = (
            df[label_column].apply(lambda x: label_2_id_dict[x])
        ).astype(int)

        df = df.convert_dtypes()
        df.drop_duplicates(inplace=True)
        df_issue_text = df[[*columns, label_column]]
        df_issue_text.dropna(inplace=True)    

        x = [
            "\n".join(tup)
            for tup in zip(*[list(df_issue_text[col]) for col in columns])
        ]

        y = list(df_issue_text[label_column])

        return x, y

    def load_tokenizer(self, model_name: str = None) -> RobertaTokenizer:
        """
            Create & configure tokenizer from hub
            Parameters:
                model_name : str = (optional) name of the model whose tokenizer will be loaded, defaults to internally configured one
            Return:
                RobertaTokenizer: tokenizer to retrieve
        """
        return AutoTokenizer.from_pretrained(
            model_name or self._base_model_name, id2label=self.labelset.get_id2label_dict()
        )

    def load_base_model(
        self, model_name: str = None, num_labels : int = None
    ) -> RobertaForSequenceClassification:
        """
            Create model from model hub, configure them and retrieve it
            Parameters:
                model_name : str = (optional) name of the model to load
            Return:
                RobertaForSequenceClassification : the model as specified
        """
        # Creating model and tokenizer
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name or self._base_model_name, num_labels=num_labels or self.labelset.num_labels()
        )
        # Use GPU if available
        model.to(self._device)

        return model

    def transform_dataset(
        self, x: List[str], y: List[int], tokenizer: RobertaTokenizer,
    ) -> Dataset:
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

        X_tokenized = tokenizer(x, padding=True, truncation=True, max_length=512)

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

        dataset = _Dataset(X_tokenized, y)

        return dataset

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
            "output_dir": self.results_path,
            "num_train_epochs": 1,
            "per_device_train_batch_size": 10,
            "per_device_eval_batch_size": 10,
            "warmup_steps": 500,
            "weight_decay": 0.01,
            "logging_dir": self.logs_path,
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
                train_dataset: Dataset = Optional dataset to use during training, overriding the default one
                eval_dataset: Dataset = Optional dataset to use during training, overriding the default one
                

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

        model.save_pretrained(path_to_save_checkpoint or self.files_folder_path)

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
            path, local_files_only=True, id2label=self.labelset.get_id2label_dict()
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
        columns: List[str] = ["content"],
        training_dataset: str = "classifier_training_dataset.csv",
        confirmation_dataset: str = "classifier_confirmation_dataset.csv",
        val_test_proportion: float = 0.2,
        base_model_name: str = None,
        labelset: Type[LabelSet] = None,
        label_column: str = None
    ) -> DataFrame:
        """
            Run an experiment specified by given train_args, and write a summary if requested so
            Parameters:
                train_args : Dict[str, Any] = (optional) arguments passed to trainig arguments
                columns : [str] = (optional) columns to use in the dataset
                training_dataset : str = (optional) dataset name to use during training, should be a name of a dataset under <project_root>/data/raw/huggingface
                confirmation_dataset : str = (optional) dataset name to use after training as confirmation dataset
                val_test_proportion : float = (optional) how much proportion of the training dataset to use as validation dataset
                base_model_name : str = (optional) Name of the model to use as a base for this experiment, defaults to the stored one if not provided
            Return:
                Classifier metrics
        """

        # Sanity checks
        assert (
            0.0 < val_test_proportion < 1.0
        ), "val_test_proportion should be in range (0,1)"
        assert all(
            c in [f.name for f in dataclasses.fields(ScrapedData)] for c in columns
        ), "columns should be valid ScrapedData fields"

        # check that you have a folder where to store results
        if not self.files_folder_path:
            raise ValueError(
                "Can't train in a Classifier without a folder for local data"
            )

        # Prepare training dataframe and load model + tokenizer
        x, y = self.prepare_dataframe(
                        columns=columns, 
                        dataset_name=training_dataset, 
                        label_column=label_column, 
                        labelset=labelset
                    )

        # Split dataset into training and validation
        X_train, X_val, y_train, y_val = train_test_split(
            x, y, test_size=val_test_proportion, stratify=y
        )

        # Load model and tokenizer
        model = self.load_base_model(model_name=base_model_name)
        tokenizer = self.load_tokenizer(model_name=base_model_name)

        # transform data into datasets
        train_dataset = self.transform_dataset(X_train, y_train, tokenizer)
        val_dataset = self.transform_dataset(X_val, y_val, tokenizer)
        del X_train, X_val, y_train, y_val

        # Fine tune the model
        fine_tuned_model_trainer = self.train_and_save_model(
            model=model,
            output_dir=self.results_path,
            logging_dir=self.logs_path,
            path_to_save_checkpoint=self.files_folder_path,
            train_args=self._override_train_args(train_args or {}),
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
        )

        del train_dataset, val_dataset

        # Save tokenizer too
        tokenizer.save_pretrained(self.files_folder_path)

        # Prepare confirmation dataframe
        x_confirmation, y_confirmation = self.prepare_dataframe(
            columns=columns, dataset_name=confirmation_dataset
        )
        confirmation_dataset = self.transform_dataset(
            x_confirmation, y_confirmation, tokenizer
        )

        # Get the metrics from the model
        metrics_df = self.evaluate_metrics(
            trainer=fine_tuned_model_trainer, val_dataset=confirmation_dataset
        )

        return metrics_df

    def classify(
        self, data: List[ScrapedData], model: str = None
    ) -> List[Dict[str, Any]]:
        """
            Classify the given data instance, returning classification metrics
            as a simple dict.
            Parameters:
                data : ScrapedData = Data instance to classify
                model : str = model name of model to load use when classifying. If no model provided,
                              use the model configured for this classifier
            Return:
                A List of dicts with the resulting scraped data correctly labelled
                and its corresponding scores tensor for each possible label. Available fields:
                    + data : ScrapedData = resulting data instance after classification
                    + scores : torch.Tensor = Scores for each label returned by the classifier
        """
        # Get model from experiment:
        if model is None:
            model = self._files_folder
            # Check if there's a config.json in this folder
            if not Path(model, "config.json").exists():
                raise ValueError(f"path: {model} does not contains valid model to load")

        roberta_model = self.load_fine_tuned_model(model)

        # Tokenize input
        roberta_tokenizer = self.load_tokenizer()
        tokenized_input = roberta_tokenizer(
            [self._get_text_from_scrapeddata(d) for d in data],
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )

        raw_output = roberta_model(**tokenized_input)
        output: torch.Tensor = torch.nn.functional.softmax(raw_output.logits, dim=-1)

        result = []
        for (x, d) in zip(output, data):
            d.label_relevance = RelevanceClassificationLabels(self.index_to_label(torch.argmax(x).item()))
            result.append({"data": d, "scores": x})

        return result

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
        tokenizer = self.load_tokenizer()

        # Create explainer
        explainer = SequenceClassificationExplainer(model, tokenizer)

        scores = explainer(sentence, class_name=additional_label)
        label = explainer.predicted_class_name

        # write html file
        if html_file:
            explainer.visualize(html_file)

        return {"scores": scores, "label": label}

    def index_to_label(self, index: int) -> LabelSet:
        """
            Get index for label
        """
        d = self.labelset.get_id2label_dict()
        label = d.get(index)
        assert label, "Label shouldn't  be None"
        return label

    def _get_text_from_scrapeddata(self, scraped_data : ScrapedData, columns : List[str] = ["title"]) -> str:
        return ". ".join([scraped_data.__getattribute__(attr) for attr in columns])

    @classmethod
    def relevance(cls, **kwargs):
        kwargs["labelset"] = RelevanceClassificationLabels
        kwargs["label_column"] = "label_relevance"
        return cls(**kwargs)
        
