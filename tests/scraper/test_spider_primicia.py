import os
from c4v.scraper.tests.utils import fake_response_from_file
from c4v.scraper.scrapers.scraper_primicia import ScraperPrimicia
from c4v.scraper.settings import C4V_ROOT_DIR


def test_parse_ok(get_body_for_parse_ok_primicia):
    """
    Check that ScraperPrimicia parses a valid page as expected
    """
    url_path = "resources/tests/html_bodies/primicia/calle-de-los-frenos-llena-de-basura.html"
    primicia_url = "https://primicia.com.ve/guayana/ciudad/calle-de-los-frenos-llena-de-basura/"
    test_file = os.path.join(C4V_ROOT_DIR, url_path)
    response = fake_response_from_file(
        test_file,
        primicia_url,
    )

    scraper = ScraperPrimicia()
    parse_output = scraper.parse(response)

    assert parse_output["body"] == get_body_for_parse_ok_primicia["body"], "body does not match"
    assert parse_output["title"] == get_body_for_parse_ok_primicia["title"], "title does not match"
    assert parse_output["date"] == get_body_for_parse_ok_primicia["date"], "title does not match"
    assert parse_output["author"] == get_body_for_parse_ok_primicia["author"], "author does not match"
    assert set(parse_output["tags"]) == set(get_body_for_parse_ok_primicia["tags"]), "tags does not match"
