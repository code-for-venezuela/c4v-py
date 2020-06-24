import os, sys
from datetime import datetime
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
from scipy.sparse import lil_matrix

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

    def get_predictions(self, x_test1):

        if self.type_ in ['knn', 'logistic']:
            x_test = lil_matrix(x_test1).toarray()
        else:
            x_test = x_test1

        return self.model_x.predict(x_test)

    def get_metrics(self, x_test1, y_test1) -> dict:
        if self.type_ == 'cc':
            y_test = y_test1.filter(self.selected_labels, axis=1)
        else:
            y_test = y_test1

        y_pred = self.get_predictions(x_test1)

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
        # print('getting metris of ', metrics)
        metrics['model_name'] = self.model_name
        metrics['model_class_name'] = self.model_x.__class__.__name__
        return metrics


def analysis_miss_classify_tweets(model_x_list: list, data: DataLoader):
    for model in model_x_list:
        # TODO: implement here the analysis
        #  gather from each model the records that have not been correctly classified
        print(model.model_name)
        print('which ones have been wrongly predicted??')
        # print(model.get_predictions(data.X_test))
        # print(data.y_test)


def report_demo(file_names: list) -> pd.DataFrame:
    '''
    :param file_names: list of file paths that contain the annotated data and the test data (tweets)
    :return: a pandas data frame object with the report on each of the models evaluated
    '''
    # loading the data
    random_state = 42
    data = DataLoader(file_names, binary=True, )
    print(f'the shape of my data.X: {data.X.shape}')
    data.preprocess(random_state=random_state)

    # Creating the models
    br_BernoulliNB_classifier = BinaryRelevance(BernoulliNB())
    lp_BernoulliNB_classifier = LabelPowerset(BernoulliNB())
    cc_BernoulliNB_classifier = ClassifierChain(BernoulliNB())
    br_LogisticR_classifier = BinaryRelevance(LogisticRegression(random_state=random_state))
    lp_LogisticR_classifier = LabelPowerset(LogisticRegression(random_state=random_state))
    cc_LogisticR_classifier = ClassifierChain(LogisticRegression(random_state=random_state))
    br_SVM_classifier = BinaryRelevance(LinearSVC(random_state=random_state))
    lp_SVM_classifier = LabelPowerset(LinearSVC(random_state=random_state))
    cc_SVM_classifier = ClassifierChain(LinearSVC(random_state=random_state))
    br_GradDesc_classifier = BinaryRelevance(SGDClassifier(random_state=random_state, loss="log", penalty="elasticnet"))
    lp_GradDesc_classifier = LabelPowerset(SGDClassifier(random_state=random_state, loss="log", penalty="elasticnet"))
    cc_GradDesc_classifier = ClassifierChain(SGDClassifier(random_state=random_state, loss="log", penalty="elasticnet"))
    ml_classifier = MLkNN(k=4)

    # Creating a list of model wrappers. Training happens after each wrapper is instantiated.
    models = [
        Model_X(ml_classifier, data.X_train, data.y_train, name='Multi kNN', type='knn'),
        Model_X(br_BernoulliNB_classifier, data.X_train, data.y_train, name='BR_Bayes BernoulliNB'),
        Model_X(br_LogisticR_classifier, data.X_train, data.y_train, name='BR_LogisticR'),
        Model_X(br_SVM_classifier, data.X_train, data.y_train, name='BR_LinearSVC'),
        Model_X(br_GradDesc_classifier, data.X_train, data.y_train, name='BR_GradDesc SGDCClassifier'),

        Model_X(lp_BernoulliNB_classifier, data.X_train, data.y_train, name='LP_Bayes BernoulliNB'),
        Model_X(lp_LogisticR_classifier, data.X_train, data.y_train, name='LP_LogisticR'),
        Model_X(lp_SVM_classifier, data.X_train, data.y_train, name='LP_LinearSVC'),
        Model_X(lp_GradDesc_classifier, data.X_train, data.y_train, name='LP_GradDesc SGDCClassifier'),

        Model_X(cc_BernoulliNB_classifier, data.X_train, data.y_train, name='CC_Bayes BernoulliNB', type='cc'),
        Model_X(cc_LogisticR_classifier, data.X_train, data.y_train, name='CC_LogisticR', type='cc'),
        Model_X(cc_SVM_classifier, data.X_train, data.y_train, name='CC_LinearSVC', type='cc'),
        Model_X(cc_GradDesc_classifier, data.X_train, data.y_train, name='CC_GradDesc SGDCClassifier', type='cc')

    ]

    # TODO: add analysis about tweets that were not correctly identified
    analysis_miss_classify_tweets(models, data)

    # getting report
    report = pd.DataFrame([m.report_as_dict(data.X_test, data.y_test) for m in models])
    
    # TODO: Ver como hacer el logging en data science antes de implementar esto!
    # Storing on disk the csv corresponding to the report, a timestamp is added as part of the file's name.
    # the name of the original annotation file(s) will be saved on the csv with "-source" as suffix.

    # save_report_on_disk(data.source_as_list(), report)

    return report


def save_report_on_disk(file_names: list, df: pd.DataFrame) -> None:
    out_filename = f'report-{datetime.now().strftime("%Y_%m_%d_%H%M%S")}'
    df.to_csv(os.path.join('reports', out_filename+'.csv'))
    pd.DataFrame({'source': file_names}).to_csv(os.path.join('reports', out_filename+'-sources.csv'))


def missed_tweets(with_annotation: str) -> None:
    """
    Shows tweets that have not been captured by the load_data class.
    :param with_annotation: path from current location to the annotated pair .txt and .ann
     no extension required.
    :return: None
    """
    with open(f'{with_annotation}.txt') as f:
        lines = f.read().replace('\n\n', '\n').split('\n')[:-1]

    # annotated_data = pd.read_csv('helper_functions/marianela_39.csv')['text'].to_list()
    annotated_data = DataLoader([with_annotation]).X
    o_lines = [line.replace('\n', '') for line in annotated_data]

    missed_values = [line for line in lines if line not in o_lines]
    print('me faltan:', len(missed_values))
    for m in missed_values:
        print('*\t', m, '\n')


if __name__ == '__main__':
    '''
    This line needs to be added in the jupyter notebook to make use of the classes
    '''

    # -------------

    # Show me the performance of the models using annotated data

    r = report_demo(file_names=[
        # '../brat-v1.3_Crunchy_Frog/data/first-iter/balanced_dataset_brat',
        '../brat-v1.3_Crunchy_Frog/data/second-iter/diego-sample_30-randstate_19-2020-06-15_202334',  # binary false.why?
        # '../brat-v1.3_Crunchy_Frog/data/second-iter/marianela-sample_50-randstate_42-2020-06-13_195818',
        # '../brat-v1.3_Crunchy_Frog/data/first-iter/sampled_58_30'
    ])
    print(r.sort_values(by='accuracy'))

    # -------------

    # shows me which tweets were ignored by the DataLoader class

    # missed_tweets('../brat-v1.3_Crunchy_Frog/data/second-iter/marianela-sample_50-randstate_42-2020-06-13_195818')
    # missed_tweets('../brat-v1.3_Crunchy_Frog/data/second-iter/diego-sample_30-randstate_19-2020-06-15_202334')

