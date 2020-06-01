def cleaner(df,text_col, is_pandas_series = False):
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
        df[text_col] = df[text_col].str.lower()

      # Convert common spanish accents

        df[text_col] = df[text_col].str.replace("ú", "u")
        df[text_col] = df[text_col].str.replace("ù", "u")
        df[text_col] = df[text_col].str.replace("ü", "u")
        df[text_col] = df[text_col].str.replace("ó", "o")
        df[text_col] = df[text_col].str.replace("ò", "o")
        df[text_col] = df[text_col].str.replace("í", "i")
        df[text_col] = df[text_col].str.replace("ì", "i")
        df[text_col] = df[text_col].str.replace("é", "e")
        df[text_col] = df[text_col].str.replace("è", "e")
        df[text_col] = df[text_col].str.replace("á", "a")
        df[text_col] = df[text_col].str.replace("à", "a")
        df[text_col] = df[text_col].str.replace("ñ", "gn")

        # Remove Punctuation
        df[text_col] = df[text_col].str.replace("[\.\-:,\?]", " ")

        # Remove links
        df[text_col] = df[text_col].str.replace("http.+", " ")
        

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
