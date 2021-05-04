from scrapy import http
from scrapy.http import Request, Response, HtmlResponse
import importlib.resources as resources

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
