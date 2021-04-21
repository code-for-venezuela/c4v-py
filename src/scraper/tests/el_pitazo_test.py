# TODO move tests out this module when approved.

from scraper.tests.utils import fake_response_from_file
from scraper.scrapers.el_pitazo_scraper import ElPitazoScraper
from scraper.settings import ROOT_DIR
import os 


def test_title_parse_ok():
    url = "tests/html_bodies/el_pitazo_fallas_electricas_carne.html"
    test_file = os.path.join(ROOT_DIR, url)
    response = fake_response_from_file(test_file, "https://elpitazo.net/cronicas/la-fallas-electricas-han-disminuido-la-cantidad-de-carne-que-consume-el-venezolano/")

    scraper = ElPitazoScraper()
    parse_output = scraper.parse(response)

    # body = ????? TODO still don't know how to test body equality

    assert parse_output['title'] == "Las fallas eléctricas han disminuido la cantidad de carne que consume el venezolano", "title does not match"
    assert parse_output['author'] == "Redacción El Pitazo", "author does not match"
    assert parse_output['tags'] == [], "tags does not match"
    assert set(parse_output['categories']) == set(["Crónicas", "Regiones"]), "categorias no coinciden"
    # assert body ???? 