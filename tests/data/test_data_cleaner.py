from c4v.data.data_cleaner import DataCleaner
import pandas as pd


FULL_TEXT_LABEL = "full_text"
PATH_TO_SAMPLE_CORPUS = "../../data/raw/tweets/tagging-set-original_for_jupyter_tagging.csv"


def test_set_lowercase():
    sentence_to_convert = "La MaRaCuchA"
    sentence_converted = "la maracucha"
    assert DataCleaner.set_lowercase(sentence_to_convert) == sentence_converted


def test_convert_common_spanish_accents_n_tilde():
    sentence_to_convert = "úùüóòíìéèáàñ"
    sentence_converted = "uuuooiieeaagn"
    assert DataCleaner.convert_common_spanish_accents_n_tilde(sentence_to_convert) == sentence_converted

def test_convert_common_spanish_accents_n_tilde2():
    sentence_to_convert = "Tú, ven acá.  Mamà lleva un acento dìferente.  La niña se cayó en éste lugar porque aquí esta el pelón. Très chigüires, brùjula francesa!"
    sentence_converted = "Tu, ven aca.  Mama lleva un acento diferente.  La nigna se cayo en este lugar porque aqui esta el pelon. Tres chiguires, brujula francesa!"
    assert DataCleaner.convert_common_spanish_accents_n_tilde(sentence_to_convert) == sentence_converted


def test_remove_emojis():
    sentence_to_convert = ""
    sentence_converted = ""
    assert DataCleaner.remove_emojis(sentence_to_convert) == sentence_converted


def test_remove_newlines():
    sentence_to_convert = "asi \n sera "
    sentence_converted = "asi   sera "
    assert DataCleaner.remove_newlines(sentence_to_convert) == sentence_converted


def test_remove_mentions():
    sentence_to_convert = "La culpa es   de @CorpoElec."
    sentence_converted = "La culpa es   de MENTION."
    assert DataCleaner.remove_mentions(sentence_to_convert, replace_with_blank=False) == sentence_converted


def test_remove_mentions2():
    sentence_to_convert = "La culpa es   de @CorpoElec."
    sentence_converted = "La culpa es   de  ."
    assert DataCleaner.remove_mentions(sentence_to_convert) == sentence_converted


def test_remove_hashtags():
    sentence_to_convert = "#sinLuz"
    sentence_converted = "HASHTAG"
    assert DataCleaner.remove_hashtags(sentence_to_convert, replace_with_blank=False) == sentence_converted


def test_remove_hashtags2():
    sentence_to_convert = "#sinLuz"
    sentence_converted = " "
    assert DataCleaner.remove_hashtags(sentence_to_convert) == sentence_converted


def test_remove_urls():
    sentence_to_convert = "el link es: https://www.geeksforgeeks.org/python-operators/?v=1&v=2&name=Diego%20Gimenez y ya no hay mas links"
    sentence_converted = "el link es: LINK y ya no hay mas links"
    assert DataCleaner.remove_urls(sentence_to_convert, replace_with_blank=False) == sentence_converted


def test_remove_urls2():
    sentence_to_convert = "el link es: https://www.geeksforgeeks.org/python-operators/?v=1&v=2&name=Diego%20Gimenez y ya no hay mas links"
    sentence_converted = "el link es:   y ya no hay mas links"
    assert DataCleaner.remove_urls(sentence_to_convert) == sentence_converted


def test_remove_some_punctuation():
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
    sentence_to_convert = "Punto.    Guion pequeño-    Dos puntos:    Coma,    Interrogación cerrando?"
    sentence_converted = "Punto     Guion pequeño     Dos puntos     Coma     Interrogación cerrando "
    assert DataCleaner.remove_some_punctuation(sentence_to_convert) == sentence_converted


def test_remove_extra_spaces():
    sentence_to_convert = "Hay muchos       espacios    entre     las palabras     "
    sentence_converted = "Hay muchos espacios entre las palabras "
    assert DataCleaner.remove_extra_spaces(sentence_to_convert) == sentence_converted


def test_trim():
    sentence_to_convert = "  La cosa es asi  "
    sentence_converted = "La cosa es asi"
    assert DataCleaner.trim(sentence_to_convert) == sentence_converted


def test_show_cleaned_data():

    raw_data = pd.read_csv(PATH_TO_SAMPLE_CORPUS)
    cleaned_data = DataCleaner.data_prep_4_vocab(raw_data[FULL_TEXT_LABEL], replace_with_blank=False)

    # show the tweets after they have been "cleaned"
    for tweet, clean_tweet in zip(
        raw_data[FULL_TEXT_LABEL].to_list(), cleaned_data.to_list()
    ):
        print("---<START>")
        print("\t", tweet)
        print("\t", clean_tweet)
        print("<END>---")
