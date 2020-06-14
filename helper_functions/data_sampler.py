import pandas as pd
import sys
import csv
from datetime import datetime
import getopt
import argparse

class data_sampler:

    def __init__(self, tweets_dataset_path: str, sample_n:int, 
                random_state:int, text_col:str, annotator_name:str, 
                ):

        self.tweets_dataset_path = tweets_dataset_path
        # self.brat_data_path = brat_data_path
        self.sample_n = sample_n
        self.random_state = random_state
        self.text_col = text_col
        self.annotator_name = annotator_name
        self.get_sample()

    def cleaner(self, df,
     is_pandas_series = False):

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

        if is_pandas_series == False:
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
        
        elif is_pandas_series == True:
            
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

        original_df = original_df.sample(random_state = self.random_state, 
                               n = self.sample_n)[['id',self.text_col]]

        original_df = self.cleaner(df = original_df)
        
        original_df = original_df[self.text_col].str.replace('\n','') 

        original_df = original_df.apply(lambda x: x.encode('ascii', 'ignore').decode('ascii'))
        

        # Returns csv with annotator's name (e.g. Juanito perez) and with timestamp
        

        timestamp = datetime.now()
        formatted_timestamp = timestamp.strftime('%Y-%m-%d_%H:%M:%S')
        filename = f'tobrat-{self.annotator_name}-sample_n_{self.sample_n}-random_state_{self.random_state}---{formatted_timestamp}.csv'
        original_df.to_csv(f'../data/data_to_annotate/{self.annotator_name}/{filename}',
                            sep = ' ',
                            header = False,
                            index = False,
                            line_terminator = '\n\n',
                            quoting = csv.QUOTE_NONE, escapechar = ' ')

        ## To implement: Incorporate touch the .ann file in the data directory

        # !self.brat_data_path/bash ann_creator.sh
         
if __name__ == '__main__':
    '''
    Example:

python data_sampler.py ../data_analysis/tagging-set-original_for_jupyter_tagging.csv 10 19 full_text diego
    '''
     
    ## command line arguments to be implemented
    # opts, args = getopt.getopt(sys.argv, 'path:sample_n:random_sample:text_col:annotator_name')
    # print(sys.argv[1:])

    def check_input():
        parser = argparse.ArgumentParser(description='Path to csv file where tweets are located')
        parser.add_argument('--path',
                            type=str,
                            help= 'Path to csv file where tweets are located')
        
        parser = argparse.ArgumentParser(description='Sample size to extract.')
        parser.add_argument('--sample_size',
                            type=int,
                            help= 'Sample size to extract.')

        parser = argparse.ArgumentParser(description='Random State to use in the models')
        parser.add_argument('--rand_state',
                            type=int,
                            help= 'Random State to us ein the models')
        
        parser = argparse.ArgumentParser(description='Column that contains raw tweets text')
        parser.add_argument('--text_col',
                            type=str,
                            help= 'Column that contains rwe tweets text')
        
        parser = argparse.ArgumentParser(description='Name of the person that is annotating')
        parser.add_argument('--annotator_name',
                            type=str,
                            help= 'Name of the person that is annotating')
        
        args = parser.parse_args()
        return vars(args)

    args = check_input()
    # print(args)


    data_sampler(tweets_dataset_path = args['path'], 
                        sample_n = args['sample_size'], 
                        random_state = args['rand_state'], 
                        text_col = args['text_col'], 
                        annotator_name = args['annotator_name'])

#  python data_sampler.py --path ../data_analysis/tagging-set-original_for_jupyter_tagging.csv --sample_n 10 --rand_state 19 --text_col full_text --annotator_name diego
