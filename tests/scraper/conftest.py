import os
import configparser
import json
import pytest

from c4v.scraper.settings import C4V_ROOT_DIR

TEST_SCRAPER_CONSTANTS = os.path.join(
    C4V_ROOT_DIR, "resources/tests/scraper/test_scraper_constants.json"
)


@pytest.fixture
def get_body_for_parse_ok_el_pitazo():
    text_of_interest = _get_json_from_file(TEST_SCRAPER_CONSTANTS)["el-pitazo"][
        "get-body-parse-ok"
    ]
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
    with open(filename, "r") as f:
        content = f.read()
    return content
