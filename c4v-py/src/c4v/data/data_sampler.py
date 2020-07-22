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
                 save_taken: bool, out_folder: str
                 ):
        """
        Creates a Data Sampler
        :param tweets_dataset_path: path of the tweets source
        :param sample_n: number of samples to extract
        :param random_state: random state
        :param text_col: name of the column with tweets
        :param annotator_name: name of the person that will be making annotations, this will also be the
            name of the folder where the data sample is stored.
        :param save_taken: a boolean that when True will store the ids of sampled tweets so future annotators
            do not extract already annotated ids
        :param out_folder: folder where all samples will be store. Each annotator should have a folder in this
            directory
        :return: None
        """

        self.tweets_dataset_path = os.path.relpath(tweets_dataset_path)

        # parent folder where .csv with tweets lives
        self.file_dir = os.path.dirname(tweets_dataset_path)
        # name of the .csv that contains all the tweets (output of sql in bigQuery)
        self.input_file = os.path.basename(tweets_dataset_path)
        # name of the .csv that contains the ids of annotated tweets
        self.tweets_annotated_ids = self.input_file.split('.')[0] + '-annotated_ids.csv'
        self.taken_ids = self.__setup_file_of_annotated_ids()

        self.save_taken = save_taken
        # self.brat_data_path = brat_data_path
        self.sample_n = sample_n
        self.random_state = random_state
        self.text_col = text_col
        self.annotator_name = annotator_name
        self.filename = None
        self.annotator_folder = os.path.dirname(f'{out_folder}{self.annotator_name}/')  # output folder

        self.get_sample()

    def __setup_data_folder_for_annotator(self, basedir):
        """
        Creates the folder of the annotator, in case it does not exist.
        """
        if not os.path.exists(basedir):
            os.makedirs(basedir)
            print(f'\n\t* A folder for "{self.annotator_name}" has been created in path: {basedir}/ ')

    def __setup_file_of_annotated_ids(self) -> pd.DataFrame:
        """
        Reads the annotated ids from the csv.  In case the file does not exist.
        Returns an empty DataFrame, otherwise returns the contents of the file.
        """
        file_with_annotated_ids = os.path.join(self.file_dir, self.tweets_annotated_ids)

        if not os.path.isfile(file_with_annotated_ids):
            return pd.DataFrame({'id': []})

        return pd.read_csv(file_with_annotated_ids)

    def __save_annotated_ids(self, new_annotated_ids):

        if self.taken_ids.shape[0] > 0:
            # if there are ids, append the new to the list.
            annotated_ids = self.taken_ids.append(new_annotated_ids.to_frame())
        else:
            # if there are no existing ids, store this ones as new ones
            annotated_ids = new_annotated_ids

        annotated_ids.to_csv(f'{self.file_dir}/{self.tweets_annotated_ids}',
                             index=False)
        print(f'File Updated: {self.file_dir}/{self.tweets_annotated_ids}')

    def __process_and_sample(self) -> pd.DataFrame:
        """
        Reads the csv with all the tweets, avoids getting a new sample with tweets that have
        already been annotated by saving a csv with all the tweet IDs that have already been tagged.
        The resulting csv will have for name the same of that original file with the suffix "-annotated_ids".

        :return: a dataframe that contains a sample of the tweets to be annotated.
        """

        original_tweets = pd.read_csv(self.tweets_dataset_path)

        # make sure the ids are of type integer and not interpreted as a float/scientific notation
        original_tweets['id'] = original_tweets['id'].astype(int)

        # drop the ids from the original_df, so when we sample we do not repeat an already tagged tweet
        available_tweets = original_tweets[~original_tweets['id'].isin(self.taken_ids['id'].values)]

        sampled_df = available_tweets.sample(random_state=self.random_state,
                                             n=self.sample_n)[['id', self.text_col]]

        # Reports the number of total tweets, the ones taken, the ones available and tha ones sampled
        print(f'Size of original file with tweets: {original_tweets.shape}')
        print(f'Size of the tweet IDs that have been taken: {self.taken_ids.shape}')
        print(f'Size of the available tweets: {available_tweets.shape}')
        print(f'Size of the sampled tweets {sampled_df.shape}')
        # print(f'sampled: {sampled_df}')

        if self.save_taken:
            self.__save_annotated_ids(sampled_df['id'])
        else:
            # prompts the user to mark the tweets as tagged
            user_response = str(input(f'''
            Do you wish to set these sampled data as tagged?
            If you select <n> or any other key, you are at risk of extracting data that you have already worked on.
            Answer (y/n): '''))

            if user_response == 'y':
                self.__save_annotated_ids(sampled_df['id'])
                print(f'''
                File has been updated, and next time you require new data, we guarantee
                you will not have repeated data.
                ''')
            else:
                print(f'''
                You are at risk of extracting data that you have already worked on.  
                To avoid this, we recommend to tag the sampled data.
                ''')

        return sampled_df

    def cleaner(self, df, is_pandas_series=False):
        """
        Helper function to do basic text cleaning operations.
        These include: Converting text to lower case, removing spanish accents,and removing links.
        -------------------------------------------------------------------------------------------
        PARAMS
            df: Dataframe or Pandas.Series object.
            text_col: String. Column to clean.
            is_pandas_series: Boolean, Optional. If df is pandas.Series

        """

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

        original_df = self.__process_and_sample()

        original_df = self.cleaner(df=original_df)

        original_df = original_df[self.text_col].str.replace('\n', '')

        original_df = original_df.apply(lambda x: x.encode('ascii', 'ignore').decode('ascii'))

        # Returns csv with annotator's name (e.g. Juanito perez) and with timestamp

        timestamp = datetime.now()
        formatted_timestamp = timestamp.strftime('%Y-%m-%d_%H%M%S')
        self.filename = f'{self.annotator_name}-sample_{self.sample_n}-randstate_{self.random_state}-{formatted_timestamp}.txt'

        try:
            # saves the tweets in txt
            original_df.to_csv(f'{self.annotator_folder}/{self.filename}',
                               sep=' ',
                               header=False,
                               index=False,
                               line_terminator='\n\n',
                               quoting=csv.QUOTE_NONE,
                               escapechar=' ')

            # creates an empty .ann to use for annotations in brat
            open(f'{self.annotator_folder}/{self.filename.split(".")[0]}.ann', 'a').close()

            print(f'''
    Success!

    Data was saved at {self.annotator_folder}/{self.filename}

    '''
                  )
            # !self.brat_data_path/bash ann_creator.sh
        except FileNotFoundError:
            print(f'''
    Error!

    There was an error with {self.annotator_folder}/{self.filename}.
    Please check that the folder in which we are saving the data does exist.
    Name of the folder: {self.annotator_folder.split('/')[-1]}

    '''
                  )


if __name__ == '__main__':
    '''
    Example:

(1)
python data_sampler.py --path ../data_analysis/tagging-set-original_for_jupyter_tagging.csv --sample_size 30 --rand_state 19 --text_col full_text --annotator_name diego

   or 

(2)
python data_sampler.py -p=../data_analysis/tagging-set-original_for_jupyter_tagging.csv -s=30 -r=19 -tc=full_text --annotator_name diego

If case the annotator wishes not to repeat extracting tweets already annotated, a flag: -t ot --taken can be added
to the command (1) or (2) above.

For more information python data_sampler.py -h 

    '''


    def annotator_folder_exists(arg_value):
        """
        See if the folder with the annotator's name has been created.
        :param arg_value: a string that represents the name of the annotator and at the same time the name
        of the folder in which the samples will be stored.
        :return: the string, if no errors are thrown
        """
        if not os.path.isdir(f'{OUTPUT_FOLDER}/{arg_value}'):
            msg = f'The folder of annotator "{arg_value}" has not been created. Please do.'
            raise argparse.ArgumentTypeError(msg)

        return arg_value


    def source_file_exists(arg_value):
        """
        See if the csv given as path exists.
        :param arg_value: a string that represents the filepath that contains all tweets. The source.
        :return: the filepath as string, if no errors are thrown
        """
        if not os.path.isfile(arg_value):
            msg = f'The provided path "{arg_value}" cannot be found. Please make sure it exists.'
            raise argparse.ArgumentTypeError(msg)

        return arg_value


    def check_input():
        parser = argparse.ArgumentParser(description='''
        Sample tweets dataframe for annotations in Brat.

        Example:
            python data_sampler.py --path ../data_analysis/tagging-set-original_for_jupyter_tagging.csv --sample_size 30 --rand_state 19 --text_col full_text --annotator_name diego
        ''')
        parser.add_argument('-p', '--path',
                            type=source_file_exists,
                            help='Path to csv file where tweets are located.')

        parser.add_argument('-t', '--taken',
                            action="store_true",
                            default=False,
                            help='Determines whether to mark the sample of tweets obtained as tagged, so future '
                                 'annotators do not repeat annotating tweets that have already been taken.')

        parser.add_argument('-s', '--sample_size',
                            type=int,
                            help='Sample size to extract.')

        parser.add_argument('-r', '--rand_state',
                            type=int,
                            help='Random State to used in the models.')

        parser.add_argument('-tc', '--text_col',
                            type=str,
                            help='Column that contains rwe tweets text.')

        parser.add_argument('--annotator_name',
                            type=annotator_folder_exists,
                            help='Name of the person that is annotating.  Also the name of the folder where '
                                 'to save the sampled data.')

        return vars(parser.parse_args())


    # folder where all annotations will lay.
    OUTPUT_FOLDER = '../data/data_to_annotate/'

    args = check_input()
    # print(args)

    sample = DataSampler(tweets_dataset_path=args['path'],
                         sample_n=args['sample_size'],
                         random_state=args['rand_state'],
                         text_col=args['text_col'],
                         annotator_name=args['annotator_name'],
                         save_taken=args['taken'], out_folder=OUTPUT_FOLDER)

