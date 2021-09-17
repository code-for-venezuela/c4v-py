"""
    The classifier requieres a base language model that might need 
    to be pre trained for Venezuelan Spanish. With this class you can
    check how good the base model is for the task and pre train it if 
    necessary
"""
# Third party imports
from torch.utils.data import Dataset, DataLoader
from transformers import EvalPrediction, AutoTokenizer, AutoModelForMaskedLM, TrainingArguments, Trainer, BatchEncoding, AdamW
from transformers.optimization import Adafactor
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
import torch
import numpy as np
import GPUtil

# Python imports 
from typing import Callable, Iterable, List, Any, Dict
import dataclasses as dc
import tempfile 
from pathlib import Path

# Local imports
from c4v.scraper.scraped_data_classes.scraped_data import ScrapedData
from c4v.classifier.base_model import BaseModel
from c4v.config import settings

class LanguageModel(BaseModel):
    """
        This is the base model that gets trained for the classifier downstream 
        task. Use this class to evaluate accuracy for a base language model, and 
        retrain its embeddings layer if needed
    """

    def create_dataset_from_scraped_data(self, data : Iterable[ScrapedData], fields : List[str] = ["content"], tokenizer : Any = None) -> BatchEncoding:
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
            raise ValueError(f"No valid field provided in 'fields' field of 'LanguageModel.create_dataset_from_scraped_data' function, provided list: {fields}. Valid fields: {list(valid_fields)}")

        # Extract text to be used
        text = [ '\n'.join([ str(d.__getattribute__(attr)) for attr in fields_to_use]) for d in data ]

        # Clean text
        text = [ "\n".join( [ s for s in [s.strip() for s in d.split("\n")] if s != ""]).replace("\n", ". ") for d in text]

        # Set up tokenizer:
        tokenizer = tokenizer or AutoTokenizer.from_pretrained(self._base_model_name)        

        # Tokenize text
        tokenized_text = tokenizer(text, max_length=512, padding="max_length", truncation=True, return_tensors="pt") # TODO deberíamos configurar esto con argumentos en la funcion

        # Valid answers & attention mask 
        labels = torch.tensor(tokenized_text['input_ids'])

        # Input ids: fed to the model as the masked version
        input_ids = labels.detach().clone()

        # generate matrix of the same size of random numbers, we're going to use it 
        # to mask some words for the model
        rand = torch.rand(input_ids.shape)

        # Create mask array
        masked_tokens_array = rand < .15

        for (_, tk_value) in tokenizer.special_tokens_map.items():

            masked_tokens_array = masked_tokens_array * (input_ids != tokenizer.vocab[tk_value])

        # Convert selected tokens to mask 
        for i in range(input_ids.shape[0]):
            # get indices of mask positions from mask array
            selection = torch.flatten(masked_tokens_array[i].nonzero()).tolist()
            input_ids[i, selection] = tokenizer.vocab[tokenizer.mask_token] # Token <mask> id

        tokenized_text['labels'] = labels
        tokenized_text['input_ids'] = input_ids

        return tokenized_text # type = transformers.tokenization_utils_base.BatchEncoding

    @staticmethod
    def _to_pt_dataset(batch : BatchEncoding) -> Dataset:
        """
            Turn batch enconding (as the one created by create_dataset_from_scraped_data) into a pytorch dataset
        """
        class _Dataset(Dataset):
            def __init__(self, encodings):
                self.encodings = encodings
            def __getitem__(self, idx):
                return {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
            def __len__(self):
                return len(self.encodings.input_ids)
        
        return _Dataset(batch)

    def eval_accuracy(self, dataset : BatchEncoding, model : Any = None) -> torch.Tensor:
        """
            Try to eval accuracy of this language model for the given dataset.
            Parameters:
                dataset : BatchEnconding = dataset with data to use for evaluation, should contain the 
                                    input ids vector to classify, with masked words
                model : Any = Huggingface model for masked language modeling, be default we use the model refered
                              by the configured model name, you can override it by providing this field
            Return:
                A tensor representing the loss when evaluating the model with the given data
        """
        # Send data to corresponding device
        dataset.to(self._device)
    
        # set up model
        model = model or AutoModelForMaskedLM.from_pretrained(self._base_model_name)       

        # Load model
        model.to(self._device)

        # Evaluate model 
        outputs = model(**dataset)
        return outputs.loss

    def train_model(self, train_dataset : Dataset, eval_dataset : Dataset, model : Any = None, batch_size : int = 16, learning_rate : float = 5e-5, epochs : int = 3):
        """
            Train a given model, or the one for the provided model 
            Parameters: 
                dataset : Dataset = training data with input_ids and labels
                model : Any = model to train, provide this field to override the configured model
        """
        # Estamos probando con un training loop custom (y no con el objeto Trainer) para ver si esto nos 
        # concede mejor control sobre la memoria, estamos teniendo muchos problemas de out of memory

        # Create a data loader
        loader = DataLoader(train_dataset, batch_size, shuffle=True)

        # Set up model
        model_to_train = model or AutoModelForMaskedLM.from_pretrained(self._base_model_name)
        model_to_train.to(self._device)
        model_to_train.train() # Go to train mode

        # Create an optimizer 
        #optim  = AdamW(model_to_train.parameters(), lr = learning_rate)
        optim = Adafactor(model_to_train.parameters(), scale_parameter=True, relative_step=True, warmup_init=True, lr=None)

        for epoch in range(epochs):
            loop = tqdm(loader, leave=True)
            for batch in loop:
                # initialize calculated gradients (from prev step)
                optim.zero_grad()
                # pull all tensor batches required for training
                # get size of each batch
                input_ids = batch['input_ids'].to(self._device)
                attention_mask = batch['attention_mask'].to(self._device)
                labels = batch['labels'].to(self._device)

                # process
                outputs = model_to_train(input_ids, attention_mask=attention_mask, labels=labels)

                del outputs.logits

                # extract loss
                loss = outputs.loss

                # calculate loss for every parameter that needs grad update
                loss.backward()

                # update parameters
                optim.step()
            
                # print relevant info to progress bar
                loop.set_description(f'Epoch {epoch}')
                loop.set_postfix(loss=loss.item())

        # Evaluation
        model_to_train.eval()                       # go to eval mode
        outputs = model(**eval_dataset.encodings)   # eval for dataset



