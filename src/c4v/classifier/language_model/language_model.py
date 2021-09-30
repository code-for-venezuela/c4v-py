"""
    The classifier requieres a base language model that might need 
    to be pre trained for Venezuelan Spanish. With this class you can
    check how good the base model is for the task and pre train it if 
    necessary
"""
# Third party imports
from torch.utils.data import Dataset
from transformers import (
    EvalPrediction,
    AutoTokenizer,
    AutoModelForMaskedLM,
    TrainingArguments,
    Trainer,
    BatchEncoding,
    PreTrainedModel,
)
import pandas as pd
import torch

# Python imports
from typing import Callable, Iterable, List, Any, Dict
import dataclasses as dc
import tempfile

# Local imports
from c4v.scraper.scraped_data_classes.scraped_data import ScrapedData
from c4v.classifier.base_model import BaseModel


class LanguageModel(BaseModel):
    """
        This is the base model that gets trained for the classifier downstream 
        task. Use this class to evaluate accuracy for a base language model, and 
        retrain its embeddings layer if needed
    """

    def create_dataset_from_scraped_data(
        self,
        data: Iterable[ScrapedData],
        fields: List[str] = ["content"],
        tokenizer: Any = None,
    ) -> BatchEncoding:
        """
            Creates a dataset from ScrapedData instances, ready to be used for training or 
            evaluation, containing both the masked version and its corresponding actual answer.

            Parameters:
                data : Iterable[ScrapedData] = Data to use in the dataset 
                fields : [str] = (optional) names of fields in each instance to use when extracting text
                tokenizer : Any = (optional) A huggingface tokenizer used to tokenize text into input ids, use the 
                                  configured model's tokenizer by default, provide this field to override it
            Return:
                A dataset with masked and correct answer based on the given data as rows
        """

        # Sanity check: provided fields are valid fields
        valid_fields = set(f.name for f in dc.fields(ScrapedData))
        fields_set = set(fields)

        # Use only fields that are both in the provided list and the valid fields list
        fields_to_use = valid_fields.intersection(fields_set)

        # Raise error if can't use any field
        if not fields_to_use:
            raise ValueError(
                f"No valid field provided in 'fields' field of 'LanguageModel.create_dataset_from_scraped_data' function, provided list: {fields}. Valid fields: {list(valid_fields)}"
            )

        # Extract text to be used
        text = [
            "\n".join([str(d.__getattribute__(attr)) for attr in fields_to_use])
            for d in data
        ]

        # Clean text
        text = [
            "\n".join(
                [s for s in [s.strip() for s in d.split("\n")] if s != ""]
            ).replace("\n", ". ")
            for d in text
        ]

        # Set up tokenizer:
        tokenizer = tokenizer or AutoTokenizer.from_pretrained(self._base_model_name)

        # Tokenize text
        tokenized_text = tokenizer(
            text,
            max_length=512,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )  # TODO deberíamos configurar esto con argumentos en la funcion

        # Valid answers & attention mask
        labels = torch.tensor(tokenized_text["input_ids"])

        # Input ids: fed to the model as the masked version
        input_ids = labels.detach().clone()

        # generate matrix of the same size of random numbers, we're going to use it
        # to mask some words for the model
        rand = torch.rand(input_ids.shape)

        # Create mask array
        masked_tokens_array = rand < 0.15

        for (_, tk_value) in tokenizer.special_tokens_map.items():

            masked_tokens_array = masked_tokens_array * (
                input_ids != tokenizer.vocab[tk_value]
            )

        # Convert selected tokens to mask
        for i in range(input_ids.shape[0]):
            # get indices of mask positions from mask array
            selection = torch.flatten(masked_tokens_array[i].nonzero()).tolist()
            input_ids[i, selection] = tokenizer.vocab[
                tokenizer.mask_token
            ]  # Token <mask> id

        tokenized_text["labels"] = labels
        tokenized_text["input_ids"] = input_ids

        return (
            tokenized_text  # type = transformers.tokenization_utils_base.BatchEncoding
        )

    @staticmethod
    def to_pt_dataset(batch: BatchEncoding) -> Dataset:
        """
            Turn batch enconding (as the one created by create_dataset_from_scraped_data) into a pytorch dataset
        """

        class _Dataset(Dataset):
            def __init__(self, encodings):
                self.encodings = encodings

            def __getitem__(self, idx):
                return {
                    key: torch.tensor(val[idx]) for key, val in self.encodings.items()
                }

            def __len__(self):
                return len(self.encodings.input_ids)

        return _Dataset(batch)

    def eval_accuracy(
        self, dataset: Dataset, model: Any = None, batch_size: int = 1
    ) -> float:
        """
            Try to eval accuracy of this language model for the given dataset.
            Parameters:
                dataset : BatchEnconding = dataset with data to use for evaluation, should contain the 
                                            input ids vector to unmask, with masked words. Shoud provide the attention_mask,
                                            inputs_ids, and labels fields
                model : Any = Huggingface model for masked language modeling, be default we use the model refered
                              by the configured model name, you can override it by providing this field
                batch_size : int = batch size to send to gpu for evaluation
            Return:
                The loss for the given dataset
        """
        # set up model
        model = model or self.model

        # Load model
        model.to(self._device)

        with tempfile.TemporaryDirectory() as temp_dir:
            args = TrainingArguments(
                per_device_eval_batch_size=batch_size,
                output_dir=temp_dir,
                per_device_train_batch_size=1,
                eval_accumulation_steps=1,
                do_train=False,
                prediction_loss_only=True,
            )
            trainer = Trainer(args=args, model=model, eval_dataset=dataset)
            # Evaluate model
            outputs = trainer.evaluate()

        return outputs["eval_loss"]

    @property
    def _default_train_args(self) -> Dict[str, Any]:
        """
            A default version of training arguments as a dict
        """

        default = {
            "output_dir": self.results_path,
            "num_train_epochs": 1,
            "per_device_train_batch_size": 10,
            "per_device_eval_batch_size": 10,
            "warmup_steps": 10000,
            "weight_decay": 0.01,
            "logging_dir": self.logs_path,
            "save_total_limit": 1,
        }

        return default

    @property
    def model(self) -> PreTrainedModel:
        """
            Internal model object. It's lazy-loaded, so it will be loaded once when it's called for the first time
        """
        if self._model == None:
            self._model = AutoModelForMaskedLM.from_pretrained(
                self._base_model_name
            )  # using specific desired model

        return self._model

    def _override_train_args(self, new_args: Dict[str, Any] = {}) -> Dict[str, Any]:
        """
            Return the default args dict, with overriden settings specified as the ones specified in 
            input dict
        """

        default = self._default_train_args
        for (k, v) in new_args.items():
            default[k] = v

        return default

    def train_and_save_model(
        self,
        train_dataset: Dataset,
        eval_dataset: Dataset,
        train_args: Dict[str, Any],
        output_dir_path: str = None,
        logging_dir_path: str = None,
        path_to_save_checkpoint: str = None,
        compute_metrics: Callable[[EvalPrediction], Dict[str, Any]] = None,
        model: Any = None,
    ) -> Trainer:
        """
            Train a given model, or the one for the provided model 
            Parameters: 
                dataset : Dataset = training data with input_ids and labels
                model : Any = model to train, provide this field to override the configured model
            Return:
                Trainer object with the fine tuned model 
        """

        if output_dir_path:
            train_args["output_dir"] = output_dir_path
        if logging_dir_path:
            train_args["logging_dir"] = logging_dir_path

        args = TrainingArguments(**self._override_train_args(train_args))

        trainer = Trainer(
            model=model,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            compute_metrics=compute_metrics,
        )

        trainer.train()

        trainer.model.save_pretrained(path_to_save_checkpoint or self.files_folder_path)

        return trainer

    def evaluate_metrics(self, trainer: Trainer, val_dataset: Dataset) -> pd.DataFrame:
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
        train_dataset: Dataset,
        eval_dataset: Dataset,
        confirmation_dataset: Dataset,
        train_args: Dict[str, Any] = None,
        model_name: str = None,
    ) -> pd.DataFrame:
        """
            Run an experiment specified by given train_args, and write a summary if requested so
            Parameters:
                train_dataset : Dataset = dataset to use during training
                eval_dataset : Dataset = dataset to use during training, when evaluating a model
                confirmation_dataset : Dataset = dataset to use after training, when validating a model
                train_args : Dict[str, Any] = (optional) arguments passed to trainig arguments
                model_name : str = (optional) Base model name for selecting next model to use
            Return:
                Classifier metrics as a Dataframe
        """

        # check that you have a folder where to store results
        if not self.files_folder_path:
            raise ValueError(
                "Can't train in a Classifier without a folder for local data"
            )

        # Prepare dataframe and load model
        assert model_name or self._base_model_name, "Model to train not provided"
        model = (
            AutoModelForMaskedLM.from_pretrained(model_name)
            if model_name
            else self.model
        )

        # Fine tune the model
        fine_tuned_model_trainer = self.train_and_save_model(
            model=model,
            output_dir_path=self.results_path,
            logging_dir_path=self.logs_path,
            path_to_save_checkpoint=self.files_folder_path,
            train_args=self._override_train_args(train_args or {}),
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
        )

        # Save tokenizer too
        self.tokenizer.save_pretrained(self.files_folder_path)

        # Get the metrics from the model
        metrics_df = self.evaluate_metrics(
            trainer=fine_tuned_model_trainer, val_dataset=confirmation_dataset
        )

        return metrics_df
