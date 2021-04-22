from sys import stderr
from scraper.utils import *
from scrapy.http import Request, HtmlResponse, request
from scraper.tests import utils


def test_strip_http_tags():
    s1 = "<p>lorelei</p>"
    s2 = "<p>lorelei  </p>"
    s3 = "<p> lorelei</p>"
    s4 = "<p> lorelei </p>"
    s5 = "<p>lorelei"
    s6 = "lorelei</p>"
    s7 = "lorelei"
    assert strip_http_tags(s1) == "lorelei"
    assert strip_http_tags(s2) == "lorelei  "
    assert strip_http_tags(s3) == " lorelei"
    assert strip_http_tags(s4) == " lorelei "
    assert strip_http_tags(s5) == "lorelei"
    assert strip_http_tags(s6) == "lorelei"
    assert strip_http_tags(s7) == "lorelei"


def test_get_text():
    fake_response = get_dummy_response()

    assert get_element_text(".c1 > p", fake_response) == "tangananina"
    assert get_element_text(".c2 > span", fake_response) == "tanganana"
    assert get_element_text("ul", fake_response) == "arepapernilmondongo"


def get_dummy_response():
    body = "<div class='c1'> <p>tangananina</p> </div> <div class='c2'> <span>tanganana</span> </div>"
    body += "<ul><li>arepa</li><li>pernil</li><li>mondongo</li></ul>"

    return utils.fake_response_from_str(body, "https://www.dummy.com")
