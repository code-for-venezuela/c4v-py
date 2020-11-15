import pandas as pd
import re

class DataCleaner:
    """
    This class will provide all methods needed for cleaning the data.
    """

    def __init__(self):
        pass

    @staticmethod
    def set_lowercase(data: pd.DataFrame or str) -> pd.DataFrame or str:
        # change all string to lowercase
        if isinstance(data, pd.DataFrame) or isinstance(data, pd.Series):
            return data.str.lower()
        elif isinstance(data, str):
            return data.lower()

    @staticmethod
    def convert_common_spanish_accents_n_tilde(
        data: pd.DataFrame or str,
    ) -> pd.DataFrame or str:
        # á é í ó ú -> aeiou and ñ -> gn
        if isinstance(data, pd.DataFrame) or isinstance(data, pd.Series):
            data = data.str.replace("ú", "u")
            data = data.str.replace("ù", "u")
            data = data.str.replace("ü", "u")
            data = data.str.replace("ó", "o")
            data = data.str.replace("ò", "o")
            data = data.str.replace("í", "i")
            data = data.str.replace("ì", "i")
            data = data.str.replace("é", "e")
            data = data.str.replace("è", "e")
            data = data.str.replace("á", "a")
            data = data.str.replace("à", "a")
            data = data.str.replace("ñ", "gn")
            return data
        elif isinstance(data, str):
            data = data.replace("ú", "u")
            data = data.replace("ù", "u")
            data = data.replace("ü", "u")
            data = data.replace("ó", "o")
            data = data.replace("ò", "o")
            data = data.replace("í", "i")
            data = data.replace("ì", "i")
            data = data.replace("é", "e")
            data = data.replace("è", "e")
            data = data.replace("á", "a")
            data = data.replace("à", "a")
            data = data.replace("ñ", "gn")
            return data

    @staticmethod
    def remove_emojis(data: pd.DataFrame or str) -> pd.DataFrame or str:
        # Ignore emojis
        if isinstance(data, pd.DataFrame) or isinstance(data, pd.Series):
            return data.apply(lambda x: x.encode("ascii", "ignore").decode("ascii"))
        elif isinstance(data, str):
            return data.encode("ascii", "ignore").decode("ascii")

    @staticmethod
    def remove_newlines(data: pd.DataFrame or str) -> pd.DataFrame or str:
        # Remove new line
        blank = "\n"
        if isinstance(data, pd.DataFrame) or isinstance(data, pd.Series):
            return data.str.replace(blank, " ")
        elif isinstance(data, str):
            return data.replace(blank, " ")

    @staticmethod
    def remove_mentions(data: pd.DataFrame or str) -> pd.DataFrame or str:
        # remove mentions
        regex = r"@[\w]+"
        if isinstance(data, pd.DataFrame) or isinstance(data, pd.Series):
            return data.str.replace(regex, "MENTION")
        elif isinstance(data, str):
            return re.sub(regex, "MENTION", data)

    @staticmethod
    def remove_hashtags(data: pd.DataFrame or str) -> pd.DataFrame or str:
        # hashtags
        regex = r"#[\w\d]+"
        if isinstance(data, pd.DataFrame) or isinstance(data, pd.Series):
            return data.str.replace(regex, "HASHTAG")
        elif isinstance(data, str):
            return re.sub(regex, "HASHTAG", data)

    @staticmethod
    def remove_urls(data: pd.DataFrame or str) -> pd.DataFrame or str:
        # remove url: http links
        regex = r"https?://[\.\w\/\-=&?%\d]+"
        if isinstance(data, pd.DataFrame) or isinstance(data, pd.Series):
            return data.str.replace(regex, "LINK")
        elif isinstance(data, str):
            return re.sub(regex, "LINK", data)

    @staticmethod
    def remove_some_punctuation(
        data: pd.DataFrame or str, all_punctuation: bool = False
    ) -> pd.DataFrame or str:
        if all_punctuation:
            #  punctuation => . - : , ? ¿ ! ¡ * } { [ ] ( ) ' " ;
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
            regex = r"[\.\-:,\?¿!¡*\}\{\[\]\(\)\'\"\;]"
        else:
            # punctuation => . - : , ?
            regex = r"[\.\-:,\?]"

        if isinstance(data, pd.DataFrame) or isinstance(data, pd.Series):
            return data.str.replace(regex, " ")
        elif isinstance(data, str):
            return re.sub(regex, " ", data)

    @staticmethod
    def remove_extra_spaces(data: pd.DataFrame or str) -> pd.DataFrame or str:
        # removes extra spaces
        regex = r"[\s]+"
        if isinstance(data, pd.DataFrame) or isinstance(data, pd.Series):
            return data.str.replace(regex, " ")
        elif isinstance(data, str):
            return re.sub(regex, " ", data)

    @staticmethod
    def trim(data: pd.DataFrame or str) -> pd.DataFrame or str:
        # spaces before and after string content.
        if isinstance(data, pd.DataFrame) or isinstance(data, pd.Series):
            return data.str.strip()
        elif isinstance(data, str):
            return data.strip()


    @staticmethod
    def data_prep_4_vocab(df: pd.DataFrame) -> pd.DataFrame:
        """
        This method provides preprocessing cleaning steps to prepare data for the vocabulary generation.
        Taking a panda DataFrame and returning the same object with all modifications applied.
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
