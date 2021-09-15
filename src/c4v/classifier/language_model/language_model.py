"""
    The classifier requieres a base language model that might need 
    to be pre trained for Venezuelan Spanish. With this class you can
    check how good the base model is for the task and pre train it if 
    necessary
"""
# Third party imports
from posixpath import basename
from torch.utils.data import Dataset
from transformers import EvalPrediction, AutoTokenizer
import torch

# Python imports 
from typing import Iterable, List, Any
import dataclasses as dc

# Local imports
from c4v.scraper.scraped_data_classes.scraped_data import ScrapedData
from c4v.config import settings

class LanguageModel:
    """
        This is the base model that gets trained for the classifier downstream 
        task. Use this class to evaluate accuracy for a base language model, and 
        retrain its embeddings layer if needed
    """

    def __init__(self, base_model_name : str = settings.default_base_language_model) -> None:
        self._base_model_name = base_model_name
        pass

    def eval_accuracy(dataset : Dataset) -> EvalPrediction:
        pass

    def create_dataset_from_scraped_data(self, data : Iterable[ScrapedData], fields : List[str] = ["content"], tokenizer : Any = None) -> Dataset:
        """
            Creates a dataset from ScrapedData instances, ready to be used for training or 
            evaluation, containing both the masked version and its corresponding actual answer.

            Parameters:
                data : Iterable[ScrapedData] = Data to use in the dataset 
                fields : [str] = names of fields in each instance to use when extracting text
                tokenizer : Any = A huggingface tokenizer used to tokenize text into input ids, use the 
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
            raise ValueError(f"No valid field provided in 'field' field of 'LanguageModel.create_dataset_from_scraped_data' function, provided list: {fields}. Valid fields: {list(valid_fields)}")

        # Extract text to be used
        text = [ '\n'.join([ str(d.__getattribute__(attr)) for attr in fields_to_use]) for d in data ]

        # Clean text
        text = [ "\n".join( [ s for s in [s.strip() for s in d.split("\n")] if s != ""]).replace("\n", ". ") for d in text]

        # Set up tokenizer:
        tokenizer = tokenizer or AutoTokenizer.from_pretrained(self._base_model_name)        

        # Tokenize text
        tokenized_text = tokenizer(data, max_length=512, padding="max_length", truncation=True) # TODO deberíamos 

        # Valid answers & attention mask 
        labels = torch.tensor(tokenized_text['input_ids'])
        mask = torch.tensor(tokenized_text['attention_mask'])

        # Input ids: fed to the model as the masked version
        input_ids = labels.detach().clone()

        # generate matrix of the same size of random numbers, we're going to use it 
        # to mask some words for the model
        rand = torch.rand(input_ids.shape)

        # Create mask array
        masked_tokens_array = rand < .15

        for (_, tk_value) in tokenizer.special_tokens_map.items():

            masked_tokens_array = masked_tokens_array * (input_ids != tokenizer.vocab[tk_value])


