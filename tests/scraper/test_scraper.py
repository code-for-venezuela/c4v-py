from c4v.scraper.scraper import _get_scraper_from_url
from c4v.scraper.scrapers.el_pitazo_scraper import ElPitazoScraper


def test_get_scraper_from_url_elpitazo():
    url1 = "https://elpitazo.net/occidente/trabajadores-de-ambulatorio-en-ciudad-ojeda-duermen-en-colchonetas-por-cortes-de-luz/"
    url2 = "https://elpitazo.net/oriente/por-falta-de-gas-trancaron-este-4may-la-avenida-gran-mariscal-de-cumana/"
    url3 = "https://twitter.com/HIDROCAPITALca/status/1124398644843700226"

    assert _get_scraper_from_url(url1) == ElPitazoScraper
    assert _get_scraper_from_url(url2) == ElPitazoScraper

    try:
        _get_scraper_from_url(url3)
    except ValueError as _:
        pass  # if value error raised, everything as expected
