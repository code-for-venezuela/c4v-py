"""
    Use this code to turn a list of scraped data into a traning-ready pandas dataframe
"""

from c4v.microscope.manager import Manager
from transformers import AutoTokenizer, AutoModelForMaskedLM, Trainer, TrainingArguments, EvalPrediction
from c4v.config import settings
import torch
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
import os
from sklearn.model_selection import train_test_split
from typing import Dict, Any



DEFAULT_DB = settings.local_sqlite_db or os.path.join(
    settings.c4v_folder, settings.local_sqlite_db_name
)

m : Manager = Manager.from_local_sqlite_db(DEFAULT_DB)

model_name = 'BSC-TeMU/roberta-base-bne'

# -- < Preparing data > --------------------------------------------------------------
# ARGUMENTS
LIMIT = 100
# Gathering data
data = [ "\n".join( [ s for s in [s.strip() for s in d.content.split("\n")] if s != ""]).replace("\n", ". ")   for d in m.get_all(limit=LIMIT, scraped=True)]

# Create tokenizer 
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Uncomment this line to get sorted tokens
# print( sorted(((k,v) for (k,v) in tokenizer.vocab.items()), key=lambda x: x[1], reverse=True))

# Create batch from known data
batch = tokenizer(data, max_length=512, padding="max_length", truncation=True)

# Create valid answers:
labels = torch.tensor(batch['input_ids'])
mask = torch.tensor(batch['attention_mask'])

# Input ids: fed to the model
input_ids = labels.detach().clone()

rand = torch.rand(input_ids.shape)

# Mask array, 1 if should be masked and is not some special token, 0 otherwise
# Special tokens: 
#   <s> : 0
#   <pad> : 1
#   </s>  : 2
#   <unk> : 3
# Mask token: 
#   <mask> : 4
# You can find all of this by checking at tokenizer.vocab
masked_tokens_array =    (rand < .15) *\
                (input_ids != 0) *\
                (input_ids != 1) *\
                (input_ids != 2) *\
                (input_ids != 3)

# convert tokens to mask 
for i in range(input_ids.shape[0]):
    # get indices of mask positions from mask array
    selection = torch.flatten(masked_tokens_array[i].nonzero()).tolist()
    input_ids[i, selection] = 4 # Token <mask> id

# -- < Creating dataset > -----------------------------------------------------------
encodings = {
    "input_ids" : input_ids, 
    "attention_mask" : mask,
    "labels" : labels
}

# Create our custom dataset class
from torch.utils.data import Dataset
class _Dataset(Dataset):
    def __init__(self, encodings):
        self.encodings = encodings
    def __len__(self):
        return self.encodings['input_ids'].shape[0]
    def __getitem__(self, i):
        # return dictionary of input_ids, attention_mask, and labels for index i
        return {key: tensor[i] for key, tensor in self.encodings.items()}

dataset = _Dataset(encodings)

# -- < Perform Evaluation > ---------------------------------------------------------
device = (
            torch.device("cuda")
            if torch.cuda.is_available()
            else torch.device("cpu")
        )
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

# Load model 
model = AutoModelForMaskedLM.from_pretrained(model_name)
model.to(device)
args = TrainingArguments(per_device_eval_batch_size=1, output_dir="./experiment", eval_accumulation_steps=1)

trainer = Trainer(
    args=args,
    model=model,
    eval_dataset=dataset,
    compute_metrics=compute_metrics # Si comentas esta línea, deja de dar out of memory
)

print(trainer.evaluate())