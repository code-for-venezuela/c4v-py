
from c4v.data.baseline_models import unannotated_tweets
from c4v.data.baseline_models import ModelAnalyzer



def test_baseline():
    '''
    The code within the __main__ might need to be added in the jupyter notebook to make use of the classes
    '''

    # -------------
    # Show me the performance of the models using annotated data
    file_names = [
        # '../brat-v1.3_Crunchy_Frog/data/first-iter/sampled_58_30',
        '../brat-v1.3_Crunchy_Frog/data/first-iter/balanced_dataset_brat',
        '../brat-v1.3_Crunchy_Frog/data/second-iter/diego-sample_30-randstate_19-2020-06-15_202334',
        '../brat-v1.3_Crunchy_Frog/data/second-iter/marianela-sample_50-randstate_42-2020-06-13_195818',
        '../brat-v1.3_Crunchy_Frog/data/second-iter/marianela-sample_50-randstate_42-2020-06-28_093100'
    ]

    # -------------
    # shows me which tweets were ignored by the DataLoader class
    for pair in file_names:
        unannotated_tweets(pair)

    # ANALYZER USAGE

    # -------------
    analyzer = ModelAnalyzer(file_names)
    print(analyzer.performance.sort_values(by='accuracy'))

    # -------------
    # Analysis of the data
    print(analyzer.analysis)

    # -------------
    # save results for further analysis, all reports will be saved on reports folder within helper_functions folder
    # uncomment the line below to save new reports
    # analyzer.save_reports()