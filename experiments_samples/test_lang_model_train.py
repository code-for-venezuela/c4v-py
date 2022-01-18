"""
    Perform a fillmask training on a base model.
    The data will be retrieved from the local storage of data.
    Use this script to perform a fillmask task over the stored data.
    You might need to scrape more data if the model couldn't find enough 
    data.
"""

from c4v.classifier.language_model.language_model import LanguageModel
from c4v.classifier.language_model.language_model_experiment import LanguageModelExperiment, LanguageModelExperimentArguments
from c4v.scraper.persistency_manager.sqlite_storage_manager import SqliteManager
from c4v.config import settings

# Get data for training
DATASET_SIZE = 1000    

CONFIRMATION_DATASET_SIZE = 500

#   Eval dataset will use what's left after removing the training dataset part
TRAIN_DATASET_SIZE = int(DATASET_SIZE*0.8)
#   Set up DB
db =  SqliteManager(settings.local_sqlite_db)

#   Set up datasets
data_for_training = list(db.get_all(limit = DATASET_SIZE + CONFIRMATION_DATASET_SIZE, scraped=True))

assert len(data_for_training) >= DATASET_SIZE + CONFIRMATION_DATASET_SIZE, "Couldn't get as much data as I need, please scrape more urls for more data"

#   Set up model
lang_model = LanguageModel()

#   Set Up data to use
FIELDS = ["title"] 
train_ds = LanguageModel.to_pt_dataset(
                lang_model.create_dataset_from_scraped_data(
                    data_for_training[:TRAIN_DATASET_SIZE], 
                    FIELDS
                )
            )
eval_ds  = LanguageModel.to_pt_dataset(
                lang_model.create_dataset_from_scraped_data(
                    data_for_training[TRAIN_DATASET_SIZE:DATASET_SIZE], 
                    FIELDS
                )
            )
confirmation_ds = LanguageModel.to_pt_dataset(
                lang_model.create_dataset_from_scraped_data(
                    data_for_training[DATASET_SIZE:DATASET_SIZE + CONFIRMATION_DATASET_SIZE], 
                    FIELDS
                )
            )
# Create experiment
args = LanguageModelExperimentArguments(
    model_name=settings.default_base_language_model,
    train_dataset = train_ds,
    eval_dataset = eval_ds,
    confirmation_dataset = confirmation_ds,
    train_args={
        "per_device_train_batch_size" : 3,
        "per_device_eval_batch_size" : 1,
        "num_train_epochs" : 1,
        "warmup_steps" : 1000,
        "save_steps" : 1000,
        "save_total_limit" : 1,
        "evaluation_strategy" : "epoch",
        "save_strategy" : "epoch",
        "load_best_model_at_end" : True,
        "eval_accumulation_steps" : 1,
        "learning_rate" : 5e-07,
        "adafactor" : True,
    }
)

exp = LanguageModelExperiment.from_branch_and_experiment("samples", "lang_model")
exp.run_experiment(args)