import pandas as pd
import os
import sys
import csv
from datetime import datetime
import getopt
import argparse


class DataSampler:

    def __init__(self, tweets_dataset_path: str, sample_n: int,
                 random_state: int, text_col: str, annotator_name: str,
                 ):
        """
        Creates a Data Sampler
        :param tweets_dataset_path: path of the tweets source
        :param sample_n: number of samples to extract
        :param random_state: random state
        :param text_col: name of the column with tweets
        :param annotator_name: name of the person that will be making annotations, this will also be the
        name of the folder where the data sample is stored.
        :return: None
        """

        self.tweets_dataset_path = tweets_dataset_path
        # self.brat_data_path = brat_data_path
        self.sample_n = sample_n
        self.random_state = random_state
        self.text_col = text_col
        self.annotator_name = annotator_name
        self.filename = None
        self.basedir = os.path.dirname(f'../data/data_to_annotate/{self.annotator_name}/')  # output folder
        # if the annotator_name folder has not been created, python can create one.
        # leave these commented until I chat with Diego
        # if not os.path.exists(basedir):
        #     os.makedirs(basedir)
        #     print(f'\n\t* A folder for "{self.annotator_name}" has been created in path: {basedir}/ ')

        self.get_sample()

    def cleaner(self, df,
                is_pandas_series=False):

        '''
        Helper function to do basic text cleaning operations. 
        These include: Converting text to lower case, removing spanish accents,and removing links.
        -------------------------------------------------------------------------------------------
        PARAMS
            df: Dataframe or Pandas.Series object. 
            text_col: String. Column to clean. 
            is_pandas_series: Boolean, Optional. If df is pandas.Series

        '''

        # to lower

        if not is_pandas_series:
            df[self.text_col] = df[self.text_col].str.lower()

            # Convert common spanish accents

            df[self.text_col] = df[self.text_col].str.replace("ú", "u")
            df[self.text_col] = df[self.text_col].str.replace("ù", "u")
            df[self.text_col] = df[self.text_col].str.replace("ü", "u")
            df[self.text_col] = df[self.text_col].str.replace("ó", "o")
            df[self.text_col] = df[self.text_col].str.replace("ò", "o")
            df[self.text_col] = df[self.text_col].str.replace("í", "i")
            df[self.text_col] = df[self.text_col].str.replace("ì", "i")
            df[self.text_col] = df[self.text_col].str.replace("é", "e")
            df[self.text_col] = df[self.text_col].str.replace("è", "e")
            df[self.text_col] = df[self.text_col].str.replace("á", "a")
            df[self.text_col] = df[self.text_col].str.replace("à", "a")
            df[self.text_col] = df[self.text_col].str.replace("ñ", "gn")

            # Remove Punctuation
            df[self.text_col] = df[self.text_col].str.replace("[\.\-:,\?]", " ")

            # Remove links
            df[self.text_col] = df[self.text_col].str.replace("http.+", " ")

            return df

        else:

            df = df.str.lower()

            # Convert common spanish accents

            df = df.str.replace("ú", "u")
            df = df.str.replace("ù", "u")
            df = df.str.replace("ü", "u")
            df = df.str.replace("ó", "o")
            df = df.str.replace("ò", "o")
            df = df.str.replace("í", "i")
            df = df.str.replace("ì", "i")
            df = df.str.replace("é", "e")
            df = df.str.replace("è", "e")
            df = df.str.replace("á", "a")
            df = df.str.replace("à", "a")
            df = df.str.replace("ñ", "gn")

            # Remove Punctuation
            df = df.str.replace("[\.\-:,\?]", " ")

            # Remove links
            df = df.str.replace("http.+", " ")

            return df

    def get_sample(self):

        original_df = pd.read_csv(self.tweets_dataset_path)

        original_df = original_df.sample(random_state=self.random_state,
                                         n=self.sample_n)[['id', self.text_col]]

        original_df = self.cleaner(df=original_df)

        original_df = original_df[self.text_col].str.replace('\n', '')

        original_df = original_df.apply(lambda x: x.encode('ascii', 'ignore').decode('ascii'))

        # Returns csv with annotator's name (e.g. Juanito perez) and with timestamp

        timestamp = datetime.now()
        formatted_timestamp = timestamp.strftime('%Y-%m-%d_%H%M%S')
        self.filename = f'{self.annotator_name}-sample_{self.sample_n}-randstate_{self.random_state}-{formatted_timestamp}.txt'

        try:
            # saves the tweets in txt
            original_df.to_csv(f'{self.basedir}/{self.filename}',
                               sep=' ',
                               header=False,
                               index=False,
                               line_terminator='\n\n',
                               quoting=csv.QUOTE_NONE,
                               escapechar=' ')

            # creates an empty .ann to use for annotations in brat
            open(f'{self.basedir}/{self.filename.split(".")[0]}.ann', 'a').close()

            print(f'''
                Success!

                Data was saved at {self.basedir}/{self.filename}

                '''
                  )
            # !self.brat_data_path/bash ann_creator.sh
        except FileNotFoundError:
            print(f'''
                    Error!

                    There was an error with {self.basedir}/{self.filename}.
                    Please check that the folder in which we are saving the data does exist.
                    Name of the folder: {self.basedir.split('/')[-1]}

                    '''
                  )


if __name__ == '__main__':
    '''
    Example:

python data_sampler.py --path ../data_analysis/tagging-set-original_for_jupyter_tagging.csv --sample_size 30 --rand_state 19 --text_col full_text --annotator_name diegoo
    '''


    def check_input():
        parser = argparse.ArgumentParser(description='''
        Sample tweets dataframe for annotations in brat.
        
        Example:
            python data_sampler.py --path ../data_analysis/tagging-set-original_for_jupyter_tagging.csv --sample_size 30 --rand_state 19 --text_col full_text --annotator_name diego
        ''')
        parser.add_argument('--path',
                            type=str,
                            help='Path to csv file where tweets are located')

        parser.add_argument('--sample_size',
                            type=int,
                            help='Sample size to extract.')

        parser.add_argument('--rand_state',
                            type=int,
                            help='Random State to used in the models')

        parser.add_argument('--text_col',
                            type=str,
                            help='Column that contains rwe tweets text')

        parser.add_argument('--annotator_name',
                            type=str,
                            help='Name of the person that is annotating')

        return vars(parser.parse_args())


    args = check_input()
    # print(args)

    sample = DataSampler(tweets_dataset_path=args['path'],
                              sample_n=args['sample_size'],
                              random_state=args['rand_state'],
                              text_col=args['text_col'],
                              annotator_name=args['annotator_name'])

#  python data_sampler.py --path ../data_analysis/tagging-set-original_for_jupyter_tagging.csv --sample_size 30
#  --rand_state 19 --text_col full_text --annotator_name diego
