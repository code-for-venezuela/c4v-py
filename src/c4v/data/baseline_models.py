import os, sys
from datetime import datetime
import pandas as pd
import numpy as np

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

from c4v.data.data_loader import BratDataLoader


class ModelWrapper:
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


class ModelAnalyzer:
    def __init__(self, file_names: list):
        self.data: BratDataLoader
        self.performance: pd.DataFrame
        self.analysis: pd.DataFrame

        self.generate_report(file_names)

    def __analysis_tweets_classification(self, model_x_list: list, data: BratDataLoader) -> pd.DataFrame:
        analysis_report = list()
        for model in model_x_list:
            expected = data.y_test
            predicted = model.get_predictions(data.X_test)
            all_tw, train_tw, test_tw = data.get_X_as_text()
            test_tw_df = test_tw.to_frame()
            predicted_df = pd.DataFrame(predicted.toarray(),
                                        columns=['pred_' + name for name in expected.columns.to_list()])

            assert_df = pd.DataFrame(predicted.toarray() == expected.to_numpy(),
                                     columns=['assert_' + name for name in expected.columns.to_list()])
            assert_df = assert_df.mask(assert_df == True, 1).mask(assert_df == False, 0)

            model_wrapper_df = pd.DataFrame([model.model_class_name for _ in range(data.X_test.shape[0])],
                                            columns=['model_class_name'])
            model_name_df = pd.DataFrame([model.model_name for _ in range(data.X_test.shape[0])],
                                         columns=['model_name'])

            end = pd.concat(
                [test_tw_df.reset_index(drop=True), expected.reset_index(drop=True), predicted_df, assert_df,
                 model_name_df,
                 model_wrapper_df], axis=1)

            analysis_report.append(end)

        return pd.concat(analysis_report)

    def generate_report(self, file_names: list) -> None:
        '''
        Generates pandas data frame objects with performance report using a different model. And also generates a
        table that from which we will be able to recognize what type of data our models do not correctly predict.
        :param file_names: list of file paths that contain the annotated data and the test data (tweets)
        '''
        # loading the data
        random_state = 42
        self.data = data = BratDataLoader(file_names, binary=True, )
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
            ModelWrapper(ml_classifier, data.X_train, data.y_train, name='Multi kNN', type='knn'),
            ModelWrapper(br_BernoulliNB_classifier, data.X_train, data.y_train, name='BR_Bayes BernoulliNB'),
            ModelWrapper(br_LogisticR_classifier, data.X_train, data.y_train, name='BR_LogisticR'),
            ModelWrapper(br_SVM_classifier, data.X_train, data.y_train, name='BR_LinearSVC'),
            ModelWrapper(br_GradDesc_classifier, data.X_train, data.y_train, name='BR_GradDesc SGDCClassifier'),

            ModelWrapper(lp_BernoulliNB_classifier, data.X_train, data.y_train, name='LP_Bayes BernoulliNB'),
            ModelWrapper(lp_LogisticR_classifier, data.X_train, data.y_train, name='LP_LogisticR'),
            ModelWrapper(lp_SVM_classifier, data.X_train, data.y_train, name='LP_LinearSVC'),
            ModelWrapper(lp_GradDesc_classifier, data.X_train, data.y_train, name='LP_GradDesc SGDCClassifier'),

            ModelWrapper(cc_BernoulliNB_classifier, data.X_train, data.y_train, name='CC_Bayes BernoulliNB', type='cc'),
            ModelWrapper(cc_LogisticR_classifier, data.X_train, data.y_train, name='CC_LogisticR', type='cc'),
            ModelWrapper(cc_SVM_classifier, data.X_train, data.y_train, name='CC_LinearSVC', type='cc'),
            ModelWrapper(cc_GradDesc_classifier, data.X_train, data.y_train, name='CC_GradDesc SGDCClassifier', type='cc')

        ]
        # getting table with all results, ready to use for micro analysis
        self.analysis = self.__analysis_tweets_classification(models, data)

        # getting report of models performance
        self.performance = pd.DataFrame([m.report_as_dict(data.X_test, data.y_test) for m in models])

    def save_reports(self):
        # TODO: Ver como hacer el logging en data science antes de implementar esto!
        # Storing on disk the csv corresponding to the report, a timestamp is added as part of the file's name.
        # the name of the original annotation file(s) will be saved on the csv with "-source" as suffix.
        self.__save_report_on_disk(self.data.source_as_list(), self.data.X.shape[0])

    def __save_report_on_disk(self, file_names: list, data_set_size: int) -> None:
        out_filename = f'report-{datetime.now().strftime("%Y_%m_%d_%H%M%S")}-{data_set_size}t'
        reports_folder = 'reports/ml_models_performance_analysis'
        self.performance.sort_values(by='accuracy').to_csv(os.path.join(reports_folder, out_filename+'-performance.csv'))

        pd.DataFrame({'source': file_names}).to_csv(os.path.join(reports_folder, out_filename+'-sources.csv'))

        self.analysis.to_csv(os.path.join(reports_folder, out_filename + '-analysis.csv'))


def unannotated_tweets(with_annotation: str) -> None:
    """
    Shows tweets that have not been captured by the load_data class. This function makes use of BratDataLoader
    :param with_annotation: path from current location to the annotated pair .txt and .ann
     no extension required.
    :return: None
    """
    with open(f'{with_annotation}.txt') as f:
        lines = f.read().replace('\n\n', '\n').split('\n')[:-1]

    # Careful here,
    # if .X were to be used after .preprocess(), it might have been converted into a vector, no longer text.
    annotated_data = BratDataLoader([with_annotation]).X
    o_lines = [line.replace('\n', '') for line in annotated_data]

    missed_values = [line for line in lines if line not in o_lines]

    print(f'**********************\nFile: {with_annotation}\nMissing: {len(missed_values)}\n')
    for m in missed_values:
        print('-\t', m, '\n')
