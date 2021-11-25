"""
    Sample for a classifier experiment
"""
from c4v.classifier.classifier_experiment import ClassifierArgs, ClassifierExperiment

args = ClassifierArgs(
    training_args={
        "per_device_train_batch_size" : 10,
        "per_device_eval_batch_size" : 1,
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
    train_dataset_name="relevance_training_dataset.csv",
    confirmation_dataset_name= "relevance_confirmation_dataset.csv",
    label_column="label_relevance",
    description="Classifier sample",
)

exp = ClassifierExperiment.from_branch_and_experiment("samples", "classifier")
exp.run_experiment(args)