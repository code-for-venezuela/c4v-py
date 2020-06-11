import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import hamming_loss
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

from skmultilearn.problem_transform import BinaryRelevance
from skmultilearn.problem_transform import ClassifierChain
from skmultilearn.problem_transform import LabelPowerset
from sklearn.naive_bayes import BernoulliNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import LinearSVC
from skmultilearn.adapt import MLkNN
from scipy.sparse import csr_matrix, lil_matrix
from datetime import datetime


from data_loader import load_data as DataLoader


class Model_X:
    def __init__(self, model, x_train, y_train, name, type='normal'):
        self.model_class_name = model.__class__
        self.model_name = name
        self.model_x = model
        self.type_ = type

        if self.type_ == 'cc':
            self.selected_labels = y_train.columns[y_train.sum(axis=0, skipna=True) > 0].tolist()
            y_train = y_train.filter(self.selected_labels, axis=1)
        if self.type_ in ['knn', 'logistic']:
            x_train, y_train = lil_matrix(x_train).toarray(), lil_matrix(y_train).toarray()

        self.model_x.fit(x_train, y_train)

    def get_predictions(self, x_test):
        return self.model_x.predict(x_test)

    def get_metrics(self, x_test, y_test) -> dict:
        if self.type_ == 'cc':
            y_test = y_test.filter(self.selected_labels, axis=1)
        if self.type_ == 'knn':
            x_test = lil_matrix(x_test).toarray()

        y_pred = self.model_x.predict(x_test)
        return {
            'accuracy': accuracy_score(y_test, y_pred),
            'hamming_loss': hamming_loss(y_test, y_pred),
            'f1_score_micro': f1_score(y_test, y_pred, average='micro'),
            'f1_score_macro': f1_score(y_test, y_pred, average='macro'),
            'precision_micro': precision_score(y_test, y_pred, average='micro'),
            'precision_macro': precision_score(y_test, y_pred, average='macro'),
            'recall_micro': recall_score(y_test, y_pred, average='micro'),
            'recall_macro': recall_score(y_test, y_pred, average='macro')
        }

    def report_as_dict(self, x_test, y_test):
        metrics = self.get_metrics(x_test, y_test)
        metrics['model_name'] = self.model_name
        metrics['model_class_name'] = self.model_x.__class__.__name__
        return metrics


def report_demo(filename: str) -> pd.DataFrame:
    '''
    ;param flename; path of the files that contain the annotated data and the test data (tweets)
    ;return; a pandas data frame object with the report on each of the models evaluated
    '''
    # loading the data
    data = DataLoader(filename, binary=True)
    data.preprocess()
    random_state = 21
    
    # creating the models
    br_BernoulliNB_classifier = BinaryRelevance(BernoulliNB())
    br_LogisticR_classifier = BinaryRelevance(LogisticRegression(random_state=random_state))
    lp_LogisticR_classifier = LabelPowerset(LogisticRegression(random_state=random_state))
    lp_SVM_classifier = LabelPowerset((LinearSVC(random_state=random_state))) 
    lp_gd_classifier = LabelPowerset(SGDClassifier(random_state=random_state, loss="log", penalty="elasticnet"))
    ml_classifier = MLkNN(k=4)
    cc_classifier = ClassifierChain(LogisticRegression(random_state=random_state))

    # Creating a list of model wrappers and training each model models
    models = [
        Model_X(br_BernoulliNB_classifier, data.X_train, data.y_train, name='BR_Bayes'),
        # Model_X(br_LogisticR_classifier, lil_matrix(data.X_train).toarray(), lil_matrix(data.y_train).toarray(), name='BR_logisticR'),
        Model_X(lp_LogisticR_classifier, data.X_train, data.y_train, type='logistic', name='LP_LogisticR'),
        Model_X(lp_SVM_classifier, data.X_train, data.y_train, name='LP_SVM'),
        Model_X(lp_gd_classifier, data.X_train, data.y_train, name='LP_GD'),
        Model_X(ml_classifier, data.X_train, data.y_train, type='knn', name='MLkNN'),
        Model_X(cc_classifier, data.X_train, data.y_train, type='cc', name='CC_LogisticR')
    ]

    # getting report
    report = pd.DataFrame([m.report_as_dict(data.X_test, data.y_test) for m in models])
    
    ## Ver como hacer el logging en data science antes de implementar esto!
    # report.to_csv(f'report-{datetime.now}.csv')
    
    return report


if __name__ == '__main__':
    '''
    This line needs to be added in the jupyter notebook to make use of the classes
    '''
    print(report_demo(filename='../brat-v1.3_Crunchy_Frog/data/first-iter/sampled_58_30'))