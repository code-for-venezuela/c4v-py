from c4v.classifier.classifier import ClassifierExperiment

# branch name, experiment name
experiment = ClassifierExperiment("testing", "first_one")

print(experiment.run_experiment(train_args={'num_train_epochs' : 3}))
