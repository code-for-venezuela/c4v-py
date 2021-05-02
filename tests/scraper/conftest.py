import configparser

import json
import pytest


@pytest.fixture
def get_body_for_parse_ok():
    filename = "../../resources/tests/scraper/test_scraper_constants.json"
    text_of_interest = _get_json_from_file(filename)['el-pitazo']['get-body-parse-ok']
    return text_of_interest


def _get_json_from_file(filename):
    f = open(filename)
    return json.load(f)


# not needed, erase
def _get_properties_from_file(filename):
    config = configparser.RawConfigParser()
    config.read(filename)
    return config


# not needed, erase
def _get_text_from_file(filename):
    with open(filename, 'r') as f:
        content = f.read()
    return content
