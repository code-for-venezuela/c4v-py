from c4v.scraper.scraped_data_classes.elpitazo_scraped_data import ElPitazoData
from .utils import fake_response_from_str
from c4v.scraper.scrapers.el_pitazo_scraper import ElPitazoScraper


def test_parse_ok(  el_pitazo_snapshot, 
                    el_pitazo_expected_body,
                    el_pitazo_expected_categories,
                    el_pitazo_expected_author,
                    el_pitazo_expected_title
                ):
    """
        Check that ElPitazoScraper parses a valid page as expected
    """
    response = fake_response_from_str(
        el_pitazo_snapshot,
        "https://elpitazo.net/cronicas/la-fallas-electricas-han-disminuido-la-cantidad-de-carne-que-consume-el-venezolano/",
    )

    scraper = ElPitazoScraper()

    parse_output : ElPitazoData = scraper.parse(response)

    assert parse_output.body == el_pitazo_expected_body, "body does not match"
    assert (
        parse_output.title
        == el_pitazo_expected_title
    ), "title does not match"
    assert parse_output.author == el_pitazo_expected_author, "author does not match"
    assert set(parse_output.categories) == set(
        el_pitazo_expected_categories
    ), "categorias no coinciden"
    