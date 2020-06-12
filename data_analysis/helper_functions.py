import pandas as pd
import numpy as np

def annotation_parser(dir_ann_file, print_grouped_annotations = False):
    '''
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
    '''
    
    
    ### Read Data
    to_parse_df = pd.read_csv(dir_ann_file, sep = '\t',header = None)

    # Rename coumns 
    to_parse_df.columns = ['id', 'annotation', 'text']

    # Remove the ID numbers to know if it's an entity (T) or Attribute (A)
    to_parse_df['id_parsed'] = to_parse_df.id.str.replace('\d', '')

    # Remove text span and IDs (T & A) from column. This columns has the name of the attributes and etitites 
    to_parse_df['annotation_parsed'] = to_parse_df.annotation.str.replace('[\dTA]', '')


    # Remove Relation tags
    # Change Relation Id to Null
    to_parse_df.id_parsed.replace('R', np.nan, inplace= True)

    # Remove nulls
    to_parse_df.dropna(subset=['id_parsed'], inplace= True)

    # Group by id_parsed, annotation parsed and count results
    df = to_parse_df[['id_parsed', 'annotation_parsed']].groupby(['id_parsed', 'annotation_parsed'], sort = True).agg({'annotation_parsed':['count']}).copy()

    # After the group by there's multi-index columns. We rename the columns to have the level that we want (count)
    df.columns = df.columns.levels[1]

    # sort_values by index. Here the trick is to also use sort_index!!
    
    if print_grouped_annotations == True:
    
        print(df.sort_values('count', ascending=False)\
            .sort_index(level=[0], ascending=[True]))
    else:
        pass
    
    return to_parse_df




def annotation_merger(path_to_txt_file, path_to_ann_file):
    '''
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

    '''

    # Read sampled data

    sampled_ann = annotation_parser(dir_ann_file = path_to_ann_file)

    # Subset Entities and rewrite dataframe
    sampled_ann = sampled_ann[sampled_ann.id_parsed == 'T']

    # Create span columns. Split by space.
    split_ann = sampled_ann.annotation.str.split(' ', expand = True)

    # Rename Columns
    split_ann.columns = ['Entities', 'first_char', 'last_char']

    # Create new columns with each annotation's text.
    split_ann['text'] = sampled_ann.loc[sampled_ann.id_parsed == 'T', 'text']

    split_ann.first_char = split_ann.first_char.astype(int)
    split_ann.last_char = split_ann.last_char.astype(int)


    with open(path_to_txt_file) as f:
        
	# only replace the break lines
        
        REPLACE_br = lambda s: s.replace("\n","\n")
        lines = map( REPLACE_br, f.readlines() )
        
        # save number line, length of the text and text without break lines: /n
        # assuming one line corresponds to a single tweet
        tuple_tweets = [(len(l), l) for l in lines if len(l) > 0]

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
