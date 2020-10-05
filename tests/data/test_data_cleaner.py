from c4v.data.data_cleaner import DataCleaner


def test_set_lowercase():
    # DataCleaner.method()
    # write the test that corresponds to the method
    assert 1+2 == 3


def test_convert_common_spanish_accents_n_tilde():
    # DataCleaner.method()
    # write the test that corresponds to the method
    assert 1+2 == 3


def test_remove_emojis():
    # DataCleaner.method()
    # write the test that corresponds to the method
    assert 1+2 == 3


def test_remove_newlines():
    # DataCleaner.method()
    # write the test that corresponds to the method
    assert 1+2 == 3


def test_remove_mentions():
    # DataCleaner.method()
    # write the test that corresponds to the method
    assert 1+2 == 3


def test_remove_hashtags():
    # DataCleaner.method()
    # write the test that corresponds to the method
    assert 1+2 == 3


def test_remove_urls():
    # DataCleaner.method()
    # write the test that corresponds to the method
    assert 1+2 == 3


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
    # DataCleaner.method()
    # write the test that corresponds to the method
    assert 1+2 == 3


def test_remove_extra_spaces():
    # DataCleaner.method()
    # write the test that corresponds to the method
    assert 1+2 == 3


def test_trim():
    # DataCleaner.method()
    # write the test that corresponds to the method
    assert 1+2 == 3


def test_data_prep_4_vocab():
    # DataCleaner.method()
    # write the test that corresponds to the method
    assert 1+2 == 3


def test_data_prep_4_annotate():
    # DataCleaner.method()
    # write the test that corresponds to the method
    assert 1+2 == 3
