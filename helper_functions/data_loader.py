from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import helper_functions
import pandas as pd
from nltk.corpus import stopwords
stop_words = set(stopwords.words('spanish'))

class load_data:

    def __init__(self, filepath: str, binary = True):
        """
        Receives filepath of Annotation File and text file.
        These files must be in the same folder.
        Run sef.preprocess() to have train test split vectorized data. 
        ----------------------------------------------------
        Params
            filepath: str. Path to text and annotation files without their file extension.
                It is assumed that both files have the same nanme.
            
            binary: bool. Create one-hot encoded labels. 
                If false, parsed_df contains the counts of each entity per document.

        Examples
            
            # Import class

            import sys
            sys.path.append('../helper_functions/')
            import data_loader
            
            
            #Load Data into class and parse dataframe
           
            test = data_loader.load_data('path/to/file')
            

            # Print parsed dataframe

            test.parsed_df.head()


            # Vectorize text and train test split
                # This method initialized following attributes:
                    # X_train, X_test, y_train, y_test, X, y
            test.preprocess()

            # Print X_train shape 
            test.X_train.shape
        """
 
        self.text_filename = filepath + '.txt'
        self.ann_filename =  filepath + '.ann'
        self.parsed_df = self.get_parsed_df()
        self.X = self.parsed_df.iloc[:, 0]
        self.y = self.parsed_df.iloc[:, 1:]
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None


    def get_parsed_df(self, binary = True) -> pd.DataFrame: 
        '''
        Parses DataFrame with helper function annotation_merger.
        Merges annotated entities and text file created from BRAT annotation package. 
        It is assumed that each text file's line correspond to a different document (e.g. Tweet). 

        Returns DataFrame with one-hot encoded labels if binary = True. 
        If binary = False, returns DataFrame with label counts. 


        '''
        self.parsed_df = helper_functions.annotation_merger(
			path_to_txt_file= self.text_filename,
                 	path_to_ann_file= self.ann_filename
			)
        
        if binary == True: 

            self.parsed_df.iloc[:, 1:] = self.parsed_df.iloc[:, 1:].mask(self.parsed_df.iloc[:,1:] > 1, 1)
        
            
            return self.parsed_df
        
        else: 

            return self.parsed_df


    def preprocess(self, test_size = 0.3, random_state = 21, to_vectorize = True, method = 'tfidf'):
        '''
        Creates Train Test Split.

        If vectorize = True, X_train, X_test, and X will be vectorized.
        If vectorize = False, X_train, X_test, and X will remain as raw_text.

        '''

        if to_vectorize is True:
            self.vectorize()
        else:
            pass

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X,
                                                                                self.y,
                                                                                test_size = 0.3,
                                                                                random_state = 21)
        
        

    def vectorize(self, method = 'tfidf', 
                  stop_words= None, 
                  strip_accents = 'unicode', 
                  analyzer = 'word',
                  ngram_range = (1,3), 
                  norm = 'l2', 
                  max_features = 10000):
        '''
        Computes specified vectorization method and stores it into self.X. 
        Do not run after preprocess method 
        '''

        if method == 'tfidf':
            
            tfidf_vectorizer = TfidfVectorizer(strip_accents=strip_accents, 
                                               analyzer=analyzer,
                                               ngram_range=ngram_range, 
                                               norm=norm,
                                               max_features = max_features)
            
            self.X = tfidf_vectorizer.fit_transform(self.X)

        else:
            print('Select a vectorization method from the available options')

