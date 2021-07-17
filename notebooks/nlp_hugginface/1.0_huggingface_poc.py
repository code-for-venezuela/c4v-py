# -*- coding: utf-8 -*-
"""
# Steps to run the script

1. Open Google Colab and select GPU runtime.
2. Upload elpitazo_positivelabels_devdataset.csv to Google Colab
3. copy the github branch that you're working on and the Experiment that you're running in line 170
4. Copy all the script to Google Colab
5. Run Script in Google Colab.


Important notes about this script: 
- The objective of this script is to showcase a complete fine tuning of a custom dataset with Transformers.
- The dataset has been adapted to a binary classification problem.
- We're using Google Colab because it's easy and free access to GPU. 
"""


!pip install transformers
!pip install datasets

from google.colab import drive

drive.mount("/content/drive")

import pandas as pd
import numpy as np
from transformers import (
    RobertaModel,
    RobertaTokenizer,
    Trainer,
    TrainingArguments,
    RobertaForSequenceClassification,
)
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from datasets import Dataset
import torch
import os

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(device)


def create_paths_googledrive(branch_name: str = None, experiment_name: str = None):

    if branch_name is None or experiment_name is None:
        raise ValueError("Branch Nome or Experiment Name not defined")

    full_path = f"/content/drive/MyDrive/sambil/{branch_name}/{experiment_name}"
    output_path = f"{full_path}/results"
    logs_path = f"{full_path}/logs"

    # Create output folder
    os.system(f"mkdir -p {output_path}")

    # Create logs folder
    os.system(f"mkdir -p {logs_path}")

    paths = {"full_path": full_path, "output_path": output_path, "logs_path": logs_path}

    return paths


# Reading Data & Data Wrangling
def prepare_dataframe() -> list:
    # Read dataset and subset binary labels.
    df_elpitazo_pscdd = pd.read_csv("elpitazo_positivelabels_devdataset.csv")
    # print(df_elpitazo_pscdd.tipo_de_evento.value_counts().to_markdown()) # Commenting out to avoid info overload
    df_elpitazo_pscdd["label"] = (
        df_elpitazo_pscdd.tipo_de_evento == "DENUNCIA FALTA DEL SERVICIO"
    ).astype(int)
    df_elpitazo_pscdd = df_elpitazo_pscdd.convert_dtypes()
    df_denuncia_texto = df_elpitazo_pscdd[["label", "text"]]
    df_denuncia_texto.dropna(inplace=True)

    X = list(df_denuncia_texto["text"])
    y = list(df_denuncia_texto["label"])

    return X, y


def load_model_tokenizer_from_hub():
    # Loading Model
    model = RobertaForSequenceClassification.from_pretrained(
        "mrm8488/RuPERTa-base", num_labels=2
    )
    tokenizer = RobertaTokenizer.from_pretrained("mrm8488/RuPERTa-base")

    # Use GPU
    model.to(device)

    return model, tokenizer


def transform_dataset(X: list, y: list, tokenizer) -> Dataset:
    # Train Test Split
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)
    X_train_tokenized = tokenizer(
        X_train, padding=True, truncation=True, max_length=512
    )
    X_val_tokenized = tokenizer(X_val, padding=True, truncation=True, max_length=512)

    # Create torch dataset
    class Dataset(torch.utils.data.Dataset):
        def __init__(self, encodings, labels=None):
            self.encodings = encodings
            self.labels = labels

        def __getitem__(self, idx):
            item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
            if self.labels:
                item["labels"] = torch.tensor(self.labels[idx])
            return item

        def __len__(self):
            return len(self.encodings["input_ids"])

    train_dataset = Dataset(X_train_tokenized, y_train)
    val_dataset = Dataset(X_val_tokenized, y_val)

    return train_dataset, val_dataset


def compute_metrics(p):
    pred, labels = p
    pred = np.argmax(pred, axis=1)

    accuracy = accuracy_score(y_true=labels, y_pred=pred)
    recall = recall_score(y_true=labels, y_pred=pred)
    precision = precision_score(y_true=labels, y_pred=pred)
    f1 = f1_score(y_true=labels, y_pred=pred)

    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}


def train_save_model(model, output_dir: str, logging_dir: str, full_path: str):

    args = TrainingArguments(
        output_dir=output_dir,  # output directory
        num_train_epochs=1,  # total # of training epochs
        per_device_train_batch_size=10,  # batch size per device during training
        per_device_eval_batch_size=10,  # batch size for evaluation
        warmup_steps=500,  # number of warmup steps for learning rate scheduler
        weight_decay=0.01,  # strength of weight decay
        logging_dir=logging_dir,  # directory for storing logs
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )

    # Train pre-trained model
    trainer.train()

    model.save_pretrained(full_path)

    return trainer


def load_fine_tuned_model(full_path: str):
    model_path = f"{full_path}/pytorch_model.bin"
    trainer = RobertaForSequenceClassification.from_pretrained(
        "/content/drive/MyDrive/sambil/poc_transformers/ruperta_binary_denunciafaltaservicio"
    )
    return trainer


def evaluate_metrics(trainer, val_dataset: Dataset, full_path: str):

    ## Evaluate Metrics
    metrics = trainer.evaluate(val_dataset)
    metrics_df = pd.DataFrame.from_dict(
        metrics, orient="index", columns=["metrics_value"]
    )

    trainer.evaluate(val_dataset)

    return metrics_df


if __name__ == "__main__":

    ## REPLACE NONE WITH branch name and experiment name
    paths = create_paths_googledrive(branch_name=None, experiment_name=None)

    # Prepare dataframe and load model + tokenizer
    X, y = prepare_dataframe()
    model, tokenizer = load_model_tokenizer_from_hub()
    train_dataset, val_dataset = transform_dataset(X=X, y=y, tokenizer=tokenizer)

    # Fine tune the model
    fine_tuned_model_trainer = train_save_model(
        model=model,
        output_dir=paths["output_path"],
        logging_dir=paths["logs_path"],
        full_path=paths["full_path"],
    )

    # Get the metrics from the model
    metrics_df = evaluate_metrics(
        trainer=fine_tuned_model_trainer,
        val_dataset=val_dataset,
        full_path=paths["full_path"],
    )
