from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import numpy as np


class BratDataLoader:

    def __init__(self, files_path: list, binary=True):
        """
        Receives filepath of Annotation File and text file.
        These files must be in the same folder.
        Run sef.preprocess() to have train test split vectorized data. 
        ----------------------------------------------------
        Params
            files_path: list. Path to text and annotation files without their file extension.
                It is assumed that both files have the same nanme.
            
            binary: bool. Create one-hot encoded labels. 
                If false, parsed_df contains the counts of each entity per document.

        Examples

	    # Append root folder
	    import sys
	    sys.path.append('../')

	    # Import libraries
	    import numpy as np
	    import pandas as pd
	    import sys
	    from src.c4v.data.data_loader import BratDataLoader

	    # Instantiate class with one dataset
	    loaded_data = BratDataLoader(['../data/processed/brat/sampled_58_30'], binary=True)

	    ## Get parsed dataframe with text and one-hot encoded responses
	    loaded_data.get_parsed_df()

	    # Create training and test sets
	    loaded_data.preprocess()

	    ## Print training and test sets
	    print(
	    f'''
	    Training set shape:
	    X_train: {loaded_data.X_train.shape}
	    y_train: {loaded_data.y_train.shape}

	    Testing set shape:
	    X_test: {loaded_data.X_test.shape}
	    y_test: {loaded_data.y_test.shape}
    ''')            

        """

        # pair_names is a list containing the paths of the annotated pair files, since .txt and .ann have the same
        #
        self.pair_names = files_path

        self.parsed_df = self.get_parsed_df()
        self.__tweets = self.parsed_df.iloc[:, 0]

        self.X = self.parsed_df.iloc[:, 0]
        # TODO: Investigate why labels "gasoline" has float instead of integer types; and "gas" contains NAN
        #  instead of 0 and floats instead of integer.  Start by looking at self.parsed_df
        self.y = self.parsed_df.iloc[:, 1:].fillna(value=0).astype(int)  # replaces all NAN by 0

        # Training (X,y) and Testing (X,y)
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        self.split_random_state = None

    def source_as_list(self) -> list:
        return self.pair_names

    def annotation_parser(self, dir_ann_file, print_grouped_annotations=False):
        """
            Helper function to parse Brat's annotation file (.ann). 
                Returns a dataframe with the following schema: 
                    | id                	| object 	|
                    | annotation        	| object 	|
                    | text              	| object 	|
                    | id_parsed         	| object 	|
                    | annotation_parsed 	| object 	|
        --------------------------------------------------------------
        
        Params:
            
            dir_ann_file: String, Default = None. 
                Path to ann file.
            
            print_grouped_annotation: Bool. Default = False. 
                Prints count of Entities and Attributes.
        """
        
        
        ### Read Data
        to_parse_df = pd.read_csv(dir_ann_file, sep='\t', header=None)

        # Rename columns
        to_parse_df.columns = ['id', 'annotation', 'text']

        # Remove the ID numbers to know if it's an entity (T) or Attribute (A)
        to_parse_df['id_parsed'] = to_parse_df.id.str.replace('\d', '')

        # Remove text span and IDs (T & A) from column. This columns has the name of the attributes and entities
        to_parse_df['annotation_parsed'] = to_parse_df.annotation.str.replace('[\dTA]', '')

        # Remove Relation tags
        # Change Relation Id to Null
        to_parse_df.id_parsed.replace('R', np.nan, inplace=True)

        # Remove nulls
        to_parse_df.dropna(subset=['id_parsed'], inplace=True)

        # Group by id_parsed, annotation parsed and count results
        df = to_parse_df[['id_parsed', 'annotation_parsed']]\
            .groupby(['id_parsed', 'annotation_parsed'], sort=True)\
            .agg({'annotation_parsed':['count']})\
            .copy()

        # After the group by there's multi-index columns. We rename the columns to have the level that we want (count)
        df.columns = df.columns.levels[1]

        # sort_values by index. Here the trick is to also use sort_index!!
        
        if print_grouped_annotations == True:
        
            print(df.sort_values('count', ascending=False).sort_index(level=[0], ascending=[True]))
        else:
            pass
        
        return to_parse_df

    def annotation_merger(self, path_to_txt_file, path_to_ann_file):
        """
        Helper function to merge text file and annotation file created from Brat. 

        The purpose of this function is to flatten the annotations in respect to the text. 
        This function only takes into account entities. Attributes and relations are not considered. 
        The output of this function is intended to serve as input for baseline Machine Learning algorithm.

        The function returns a Pandas Data Frame with the following schema:

        | # | Column     | Non-Null Count | Dtype  |   |
        |---|------------|----------------|--------|---|
        | 0 | text       | n non-null     | object |   |
        | 1 | <entity_1> | n non-null     | uint8  |   |
        | 2 | <entity_n> | n non-null     | uint8  |   |

        The first column corresponds to the text where the tags were made.
        The following <entity> columns are the count of the each entity tagged in the text.

        --------------------------------------------------------------------------------------------------

        Params

        path_to_txt_file: String, default = none.
        Complete or relative path to text file where annotations were made.

        path_to_ann_file: String, default = none.
        Complete or relative path to annotation file (.ann) where the annotations were stored. 

        """

        # Read sampled data

        sampled_ann = self.annotation_parser(dir_ann_file=path_to_ann_file)

        # Subset Entities and rewrite dataframe
        sampled_ann = sampled_ann[sampled_ann.id_parsed == 'T']

        # Create span columns. Split by space.
        split_ann = sampled_ann.annotation.str.split(' ', expand=True)

        # Rename Columns
        split_ann.columns = ['Entities', 'first_char', 'last_char']

        # Create new columns with each annotation's text.
        split_ann['text'] = sampled_ann.loc[sampled_ann.id_parsed == 'T', 'text']

        split_ann.first_char = split_ann.first_char.astype(int)
        split_ann.last_char = split_ann.last_char.astype(int)

        with open(path_to_txt_file) as f:
            
            lines = f.readlines()

            # save number line, length of the text and text without break lines: /n
            # assuming one line corresponds to a single tweet
            tuple_tweets = [(len(line), line) for line in lines if len(line) > 0]

            start, end, text_ = list(), list(), list()
            new_start = 0
            for ttw in tuple_tweets:

                # adds the length of the tweet
                start.append(new_start)
            
                # finds the location of the last character of the tweet
                end.append(new_start + ttw[0] - 1)
                text_.append(ttw[1])
                
                # gets the starting position of the next tweet
                new_start = new_start + ttw[0]

            text_df = pd.DataFrame({
                    "first_char": start,
                    "last_char": end,
                    "text": text_
                    })

        # Initialize column multi_id
        split_ann['multi_id'] = 0

        # For each row of original text 
        for i in range(text_df.shape[0]):
            
            # Match first character position of the annotation greater or equal to the first_char of the tweet
            # Match last character position of the annotation less than the last character of the complete tweet
            # This will subset the annotations on the complete span of the tweet.
            idx = split_ann[(split_ann.first_char >= text_df.loc[i,'first_char']) &\
                    (split_ann.first_char < text_df.loc[i,'last_char'])].index   

        #     Create common value for each dataframe. 
            split_ann.loc[idx, 'multi_id'] = i 
            text_df.loc[i, 'multi_id'] = i


        # Merge both dataframes and select the entities and the tweets' complete text
        merged = pd.merge(split_ann, text_df, on='multi_id').loc[:, ['Entities', 'text_y']]

        merged.columns = ['entities', 'text']
        # Create dummy variables from entities (these are the tags)
        dummies = pd.get_dummies(merged.entities)

        # Group by the complete text and reset index to flatten multi-index columns
        return pd.concat([merged['text'], dummies], axis = 1).groupby('text').sum().reset_index()    

    def get_parsed_df(self, binary = True) -> pd.DataFrame: 
        '''
        Parses DataFrame with helper function annotation_merger.
        Merges annotated entities and text file created from BRAT annotation package. 
        It is assumed that each text file's line correspond to a different document (e.g. Tweet). 

        Returns DataFrame with one-hot encoded labels if binary = True. 
        If binary = False, returns DataFrame with label counts. 


        '''

        # self.parsed_df = self.annotation_merger(path_to_txt_file=f'{self.pair_names[0]}.txt',
        #                                         path_to_ann_file=f'{self.pair_names[0]}.ann')
        #

        single_df_list = [self.annotation_merger(path_to_txt_file=f'{path}.txt', path_to_ann_file=f'{path}.ann')
                          for path in self.pair_names]

        merged_parsed_df = pd.concat(single_df_list, axis=0).reset_index(drop=True)

        if binary == True:

            merged_parsed_df.iloc[:, 1:] = merged_parsed_df.iloc[:, 1:].mask(merged_parsed_df.iloc[:, 1:] > 1, 1)
        
            return merged_parsed_df
        
        else: 

            return merged_parsed_df

    def preprocess(self, test_size = 0.3, random_state = 21, to_vectorize = True, method = 'tfidf'):
        '''
        Creates Train Test Split.

        If vectorize = True, X_train, X_test, and X will be vectorized.
        If vectorize = False, X_train, X_test, and X will remain as raw_text.

        '''

        self.split_random_state = random_state
        if to_vectorize is True:
            self.vectorize()
        else:
            pass

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X,
                                                                                self.y,
                                                                                test_size=0.3,
                                                                                random_state=random_state)

    def get_X_as_text(self) -> tuple:
        """
        Returns the input data as text, instead of vectors

        :return: A tuple with full records of input as text on the first position; records for training as text
          on the second position; and records used for testing in the third position.
        """

        train, test = train_test_split(self.__tweets, test_size=0.3, random_state=self.split_random_state)
        return self.__tweets, train, test

    def vectorize(self, method='tfidf',
                  stop_words= None,
                  strip_accents = 'unicode', 
                  analyzer = 'word',
                  ngram_range = (1,3), 
                  norm = 'l2', 
                  max_features = 10000):
        """
        Computes specified vectorization method and stores it into self.X. 
        Do not run after pre-process method
        """

        if method == 'tfidf':
            
            tfidf_vectorizer = TfidfVectorizer(strip_accents=strip_accents, 
                                               analyzer=analyzer,
                                               ngram_range=ngram_range, 
                                               norm=norm,
                                               max_features = max_features)
            
            self.X = tfidf_vectorizer.fit_transform(self.X)

        else:
            print('Select a vectorization method from the available options')

