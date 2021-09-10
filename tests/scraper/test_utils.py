from c4v.scraper.utils import *
import tests.scraper.utils as utils


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


def test_get_domain_from_url():
    expected_domain1 = "www.codeforvenezuela.org"
    expected_domain2 = "www.google.com"
    expected_domain3 = "elpitazo.net"
    url1 = "https://www.codeforvenezuela.org"
    url2 = "https://www.codeforvenezuela.org/about-us"
    url3 = "https://www.google.com/search?client=opera-gx&q=google&sourceid=opera&ie=UTF-8&oe=UTF-8"
    url4 = "https://elpitazo.net/occidente/trabajadores-de-ambulatorio-en-ciudad-ojeda-duermen-en-colchonetas-por-cortes-de-luz/"

    assert get_domain_from_url(url1) == expected_domain1
    assert get_domain_from_url(url2) == expected_domain1
    assert get_domain_from_url(url3) == expected_domain2
    assert get_domain_from_url(url4) == expected_domain3


def test_valid_url():
    url1 = "wwww.google.com"
    url2 = "https://www.google.com/search?client=opera-gx&q=cats&sourceid=opera&ie=UTF-8&oe=UTF-8"
    url3 = "elpitazo.com"
    url4 = "a"

    assert valid_url(url1) == False
    assert valid_url(url2)
    assert valid_url(url3) == False
    assert valid_url(url4) == False


def get_dummy_response():
    body = "<div class='c1'> <p>tangananina</p> </div> <div class='c2'> <span>tanganana</span> </div>"
    body += "<ul><li>arepa</li><li>pernil</li><li>mondongo</li></ul>"

    return utils.fake_response_from_str(body, "https://www.dummy.com")

def test_group_by_ok():
    l1 = [1,1,1,2,3,3,3,4]
    l1_ans = [[1,1,1], [2], [3,3,3], [4]]
    l2 = [i for i in range(5)]
    l2_ans = [[i] for i in range(5)]
    l3 = []
    l3_ans = []
    l4 = [1]
    l4_ans = [[1]]
    l5 = [1, 2, 1]
    l5_ans = [[1], [2], [1]]

    assert list(group_by(l1)) == l1_ans 
    assert list(group_by(l2)) == l2_ans 
    assert list(group_by(l3)) == l3_ans 
    assert list(group_by(l4)) == l4_ans 
    assert list(group_by(l5)) == l5_ans 

def test_chunks_generator():
    l1 = []
    l2 = [1]
    l3 = [1,1,1,1,1,1,1,1,1]
    
    assert list(generate_chunks(l1, 3))  == []
    assert list(generate_chunks(l2, 3)) == [[1]]
    assert list(generate_chunks(l3, 3)) == [[1,1,1], [1,1,1], [1,1,1] ]
    assert list(generate_chunks(l3, 4)) == [ [1,1,1,1], [1,1,1,1], [1] ]
    assert list(generate_chunks(l3, 1)) == [ [1] for _ in range(len(l3))]
