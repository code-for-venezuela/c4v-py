from c4v.classifier.classifier import ClassifierExperiment



experiment = ClassifierExperiment("testing", "first_one")

print(experiment.run(train_args={'num_train_epochs' : 3}))