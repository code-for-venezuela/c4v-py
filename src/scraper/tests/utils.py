import encodings
from scrapy.http import Request, Response, HtmlResponse


def fake_response_from_file(file : str, url : str) -> Response:
    """
        Create a new fake scrapy response based on the content of a file
    """
    
    with open(file, 'r') as file:
        return fake_response_from_str(file.read(), url)

def fake_response_from_str(body : str, url : str) -> Response:
    """
        Create a new scrapy response based on a string and some url, 
        so you can use it to test parse funcions
    """
    req = Request(url=url)
    
    response = HtmlResponse(request=req, body=  body, url=url, encoding = 'utf-8') 
    
    return response

