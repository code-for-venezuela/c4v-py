"""
    Sample for a service classifier experiment
"""
from c4v.classifier.classifier_experiment import ClassifierArgs, ClassifierExperiment
from c4v.scraper.scraped_data_classes.scraped_data import ServiceClassificationLabels
args = ClassifierArgs(
    training_args={
        "per_device_train_batch_size" : 15,
        "per_device_eval_batch_size" : 15,
        "num_train_epochs" : 3,
        "warmup_steps" : 1000,
        "save_steps" : 1000,
        "save_total_limit" : 1,
        "evaluation_strategy" : "epoch",
        "save_strategy" : "epoch",
        "load_best_model_at_end" : True,
        "eval_accumulation_steps" : 1,
        "learning_rate" : 5e-07,
        "adafactor" : True,
    },
    columns=['title'],
    train_dataset_name="service_training_dataset.csv",  # note that this is the same experiment as relevance classification but using different parameters
    confirmation_dataset_name= "service_confirmation_dataset.csv", 
    label_column="label_service", 
    description="Service sample",
    labelset=ServiceClassificationLabels

)

exp = ClassifierExperiment.from_branch_and_experiment("samples", "service_classifier")
exp.run_experiment(args)