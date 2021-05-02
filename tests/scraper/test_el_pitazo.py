import os
from c4v.scraper.tests.utils import fake_response_from_file
from c4v.scraper.scrapers.el_pitazo_scraper import ElPitazoScraper
from c4v.scraper.settings import ROOT_DIR


def test_parse_ok(get_body_for_parse_ok_el_pitazo):
    """
        Check that ElPitazoScraper parses a valid page as expected
    """
    url = "tests/html_bodies/el_pitazo_fallas_electricas_carne.html"
    test_file = os.path.join(ROOT_DIR, url)
    response = fake_response_from_file(
        test_file,
        "https://elpitazo.net/cronicas/la-fallas-electricas-han-disminuido-la-cantidad-de-carne-que-consume-el-venezolano/",
    )

    scraper = ElPitazoScraper()
    parse_output = scraper.parse(response)

    assert parse_output["body"] == get_body_for_parse_ok_el_pitazo, "body does not match"
    assert (
        parse_output["title"]
        == "Las fallas eléctricas han disminuido la cantidad de carne que consume el venezolano"
    ), "title does not match"
    assert parse_output["author"] == "Redacción El Pitazo", "author does not match"
    assert parse_output["tags"] == [], "tags does not match"
    assert set(parse_output["categories"]) == set(
        ["Crónicas", "Regiones"]
    ), "categorias no coinciden"
