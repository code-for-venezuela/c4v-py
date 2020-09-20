import pandas as pd


class DataCleaner:
    """
    This class will provide all methods needed for cleaning the data.
    """

    def __init__(self):
        pass

    @staticmethod
    def set_lowercase(df: pd.DataFrame) -> pd.DataFrame:
        # change all string to lowercase
        return df.str.lower()

    @staticmethod
    def convert_common_spanish_accents_n_tilde(df: pd.DataFrame) -> pd.DataFrame:
        # á é í ó ú -> aeiou and ñ -> gn
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
        return df

    @staticmethod
    def remove_emojis(df: pd.DataFrame) -> pd.DataFrame:
        # Ignore emojis
        return df.apply(lambda x: x.encode("ascii", "ignore").decode("ascii"))

    @staticmethod
    def remove_newlines(df: pd.DataFrame) -> pd.DataFrame:
        # Remove new line
        return df.str.replace("\n", " ")

    @staticmethod
    def remove_mentions(df: pd.DataFrame) -> pd.DataFrame:
        # mentions
        df = df.str.replace(r"@[\w]+", "MENTION")
        return df

    @staticmethod
    def remove_hashtags(df: pd.DataFrame) -> pd.DataFrame:
        # hashtags
        df = df.str.replace(r"#[\w\d]+", "HASHTAG")
        return df

    @staticmethod
    def remove_urls(df: pd.DataFrame) -> pd.DataFrame:
        # remove url: http links
        return df.str.replace(r"http.+", "LINK")

    @staticmethod
    def remove_some_punctuation(df: pd.DataFrame) -> pd.DataFrame:
        # punctuation: . - : , ?
        # TODO: QUESTION!!!
        #  Should we also remove !, opening exclamation and interrogation?  => ¿ ? and
        #  .: punto, punto
        #  final(period)
        #  ,: coma(comma)
        #  :: dos
        #  puntos(colon)
        #  ;: punto y coma(semicolon)
        #  —: raya(dash)
        #  -: guión(hyphen)
        #  « »: comillas(quotation marks)
        #  " : comillas (quotation marks)
        #  ' : comillas simples (single quotation marks)
        #  ¿ ?: principio y fin de interrogación(question marks)
        #  ¡ !: principio y fin de exclamación o admiración(exclamation points)
        #  ( ): paréntesis(parenthesis)
        #  []: corchetes, parénteses cuadrados(brackets)
        #  {}: corchetes(braces, curly brackets)
        #  *: asterisco(asterisk)
        #  ...: puntos suspensivos(ellipsis)

        # alternative regex that removes more punctuation r"[\.\-:,\?¿!¡*\}\{\[\]\(\)\'\"\;]"
        return df.str.replace(r"[\.\-:,\?]", " ")

    @staticmethod
    def remove_extra_spaces(df: pd.DataFrame) -> pd.DataFrame:
        # removes extra spaces
        return df.str.replace(r"[\s]+", " ")

    @staticmethod
    def trim(df: pd.DataFrame) -> pd.DataFrame:
        # spaces before and after string content.
        return df.str.strip()

    @staticmethod
    def data_prep_4_vocab(df: pd.DataFrame) -> pd.DataFrame:
        """
        This method provides preprocessing cleaning steps to prepare data for the vocabulary generation.
        """
        df = DataCleaner.set_lowercase(df)

        # Convert common spanish accents
        df = DataCleaner.convert_common_spanish_accents_n_tilde(df)

        # Remove links
        df = DataCleaner.remove_urls(df)

        # Remove mentions
        df = DataCleaner.remove_mentions(df)

        # Remove hashtags
        df = DataCleaner.remove_hashtags(df)

        # Remove Emojis
        df = DataCleaner.remove_emojis(df)

        # Remove newlines
        df = DataCleaner.remove_newlines(df)
        #
        # # Remove Punctuation
        df = DataCleaner.remove_some_punctuation(df)
        #
        # # Remove white spaces
        df = DataCleaner.remove_extra_spaces(df)
        #
        # # I need to remove all spaces before and after each string
        df = DataCleaner.trim(df)

        return df

    @staticmethod
    def data_prep_4_annotate(df: pd.DataFrame) -> pd.DataFrame:
        """
        This method cleans the data before going to brat annotator
        """

        # to lower
        df = DataCleaner.set_lowercase(df)

        # Convert common spanish accents
        df = DataCleaner.convert_common_spanish_accents_n_tilde(df)

        # Remove Punctuation
        df = DataCleaner.remove_some_punctuation(df)

        # Remove links
        df = DataCleaner.remove_urls(df)

        # # Remove new lines
        # df = DataCleaner.remove_newlines(df)
        #
        # # Remove Emojis
        # df = DataCleaner.remove_emojis(df)

        return df
