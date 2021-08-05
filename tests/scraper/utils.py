from dataclasses import asdict
from datetime import datetime
from pytz       import utc
from scrapy.http import Request, Response, HtmlResponse
from c4v.scraper.persistency_manager.base_persistency_manager   import BasePersistencyManager
from c4v.scraper.scraped_data_classes.scraped_data              import ScrapedData
import importlib_resources as resources

from c4v.scraper.settings import DATE_FORMAT

def fake_response_from_file(path: str, file : str,  url: str) -> Response:
    """
        Create a new fake scrapy response based on the content of a path
    """
    
    with resources.open_text(path, file) as html:
        return fake_response_from_str(html.read(), url)

def fake_response_from_str(body: str, url: str) -> Response:
    """
        Create a new scrapy response based on a string and some url, 
        so you can use it to test parse funcions
    """
    req = Request(url=url)

    response = HtmlResponse(request=req, body=body, url=url, encoding="utf-8")

    return response

def util_test_save_for(manager : BasePersistencyManager):
    """
        Check that a new element is available after it's saved
    """
    to_add = ScrapedData(url="www.my_obviously_not_valid_url.com")
    assert to_add not in list(manager.get_all()), "Dummy object to insert should not be available at this point"
    manager.save([to_add])
    assert [d for d in manager.get_all() if d == to_add] != [], "Dummy object to insert should  be available at this point"

def util_test_save_overrides_for(manager : BasePersistencyManager):
    """
        Check that save overrides old instances with the same url
    """
    to_override = ScrapedData(url="www.my_obviously_not_valid_url.com", content="old content")
    new_version = ScrapedData(url="www.my_obviously_not_valid_url.com", content="new content")

    manager.save([to_override])
    assert new_version not in list(manager.get_all()), "new version should not be in db before added"
    manager.save([new_version])
    new_list = list(manager.get_all())
    assert new_version in new_list, "new version should be the one in DB"
    assert to_override not in new_list, "old version should not be in DB"
    
def util_test_filter_scraped_urls(manager : BasePersistencyManager):
    """
        Check that ScrapedData listing works ok
    """
    sd1 = ScrapedData(url="www.url1.com")
    sd2 = ScrapedData(url="www.url2.com", last_scraped= datetime.now(tz=utc))
    sd3 = ScrapedData(url="www.url3.com")
    sd4 = ScrapedData(url="www.url4.com")

    manager.save([sd1, sd2, sd3])
    actual_set = {sd1, sd2, sd3} 
    data_set = set(manager.get_all())
    short_set = set(manager.get_all(limit=2))
    long_set  = set(manager.get_all(limit=10000))
    negative_set = set(manager.get_all(limit=-10))

    assert sd4 not in data_set              , "non added url should not be in db"
    assert actual_set == data_set           , "added set not the same as retrieved set"
    assert all(sd in actual_set for sd in short_set) and len(short_set) == 2, "should be at the most two elements in query with limit 2"
    assert long_set == data_set and long_set == actual_set, "should be at the most the 3 elements we added in query where limit is higher than actually added elements"
    assert negative_set == actual_set       , "when a negative number is passed as limit arguments, all elements should be retrieved"
    assert set(manager.get_all(scraped=True)) == {sd2}, "scraped = true should retrieve only already scraped elements"
    assert set(manager.get_all(scraped=False)) == {sd1, sd3}, "scraped = false should retrieve only non scraped urls"

def util_test_url_filtering(manager : BasePersistencyManager):
    """
        test that url filtering works properly
    """
    url1 = "www.url1.com"
    url2 = "www.url2.com"
    url3 = "www.url3.com"

    sd1 = ScrapedData(url=url1)
    sd2 = ScrapedData(url=url2, last_scraped= datetime.now(tz=utc))

    urls = [
        url1,
        url2,
        url3
    ]

    manager.save([sd1, sd2])

    assert set(manager.filter_scraped_urls(urls)) == {url1, url3}
