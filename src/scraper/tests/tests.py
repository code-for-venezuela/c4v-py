import unittest
from scraper.spiders import ElPitazoSpider
from scraper.scraper import scrape
from scrapy.http     import Request, Response


def create_fake_request(file, url):
    request = Request(url=url)
    print("aaaa")
    with open(file) as fake_body:
        content = fake_body.read()

    response = Response(
        url=url,
        request=request,
        body=content
    )            

    response.encoding = 'utf-8'

    yield response





class TestingScrapeMethod(unittest.TestCase):

    def setUp(self) -> None:

        self.original_func = ElPitazoSpider.start_requests
        self.pitazo_spider = ElPitazoSpider
    
        return super().setUp()

    def test_elpitazo_1(self):
        url = "https://elpitazo.net/occidente/vecinos-de-el-cruce-protestaron-luego-de-tres-dias-sin-electricidad/"
        

        scrape(url)

        

