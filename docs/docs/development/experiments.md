## Experiments and classification
The microscope library doesn't provides a trained model to perform classifications, but you can train one using an experiment to train a model. 

The library provides a few sample experiments, in this section you will learn how to run an experiment and create your own for a custom classification type.

## Running a sample experiment

You can find a few experiments in the `experiments_samples` folder in the root project directory. Today we'll use the `test_relevance_classifier.py` experiment to train a model for relevance classification. Let's **create a file in the root project directory** and paste the following code from such experiment:

```python
"""
    Sample for a classifier experiment
"""
from c4v.classifier.classifier_experiment import ClassifierArgs, ClassifierExperiment
from c4v.scraper.scraped_data_classes.scraped_data import RelevanceClassificationLabels
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
    labelset=RelevanceClassificationLabels

)

exp = ClassifierExperiment.from_branch_and_experiment("samples", "relevance_classifier")
exp.run_experiment(args)
```
We're going to explain step by step what's happening here:   

1. The `train_args` dictionary is a dictionary with arguments that are later passed to the huggingface training API as training arguments. More about huggingface training arguments [here](https://huggingface.co/docs/transformers/main_classes/trainer#transformers.TrainingArguments).   
2. The `columns` field tells the model which fields to use as input from the specified dataset. All our datasets should have at the least the fields in the `ScrapedData` data format. This experiment is using the title as input only.   
3. The `train_dataset_name` is the name of a dataset stored in `data/processed/huggingface` .    
4. The `label_column` field is the name of the column inside the provided dataset you will use as target label. Por example, if you want to predict results in column `A`, then this field is set to `A`.     
5. `description` is a human readable description for this experiment, you can omit it, but is highly recommended to understand what you were trying to achieve with this model.    
6. `labelet` it's an enum class specifying which labels this model should use, as this is a multilabel training.

Now, to run this experiment we just run it like any other python program with `python sample_experiment.py`

This is how usually an experiment looks like. Most experiments to improve a classifier's performance will change just this, the values passed to the model, and the previous configuration is encapsulated by the `Experiment` class.

## Creating a new experiment

To create your own experiment, you just have to implement the following classes that you can find in `src/c4v/classifier/experiment.py`:

* `BaseExperimentSummary` : It's a class that provides information about the experiment's results. You should add fields relevant to your experiment, and then override how the information is printed (by overriding the `__str__` method). This class is used to store in local storage information about the experiment.
* `BaseExperimentArguments` : It's a class that holds the necessary data for your experiment. It's important so the library can register which arguments you used for previous experiments.
* `BaseExperiment` : This class is the experiment that will be run. All you have to do is to implement your experiment setup and execution in the `experiment_to_run`. Note that such function receives a `BaseExperimentArguments` as input and returns an instance of `BaseExperimentSummary` as output. 

Now, if you implement your experiments this way, all the data related to your experiments will be managed automatically by the `c4v-py` library, and the resulting model itself will be available for classification!.

## Classifier classes
Right now, we provide support for relevance classification and service classification, but note that those are not specific classes of a model, they're the same `Classifier` model class with different configuration options. You can create them by using:

```python
# Create a relevance classifier
relevance_model = Classifier.relevance(<Same args as in the classifier class>)

# Create a service classifier
relevance_model = Classifier.service(<Same args as in the classifier class>)
```

This is the desired way to create multilabel classifiers. For a fine grained implementation, you should inherit the `BaseModel` class in `src/c4v/classifier/base_model.py`