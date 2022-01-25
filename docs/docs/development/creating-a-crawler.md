# Creating a Crawler
To create a crawler, all you need to do is to implement the base class `BaseCrawler` in `c4v.scraper.crawler.crawlers.base_crawler`. To do so, you have to provide an implementation for the required methods, a crawler `name`, and its `start_sitemap_url`. 

For example:
```python
from c4v.scraper.crawler.crawlers.base_crawler import BaseCrawler

class SampleCrawler(BaseCrawler):
    
    start_sitemap_url = "https://samplesite.com/sitemap.xml"
    name = "sample"
```

1. `start_sitemap_url` is the **site's sitemap**, which can be often found in `<your site domain>/sitemap.xml`
2. `name` is the crawler's name, is **necessary to refer to it** on multiple operations

Now we have to implement the **static method**
``should_crawl``
```python
class SampleCrawler(BaseCrawler):
    
    # crawler name and start sitemap url...

    @staticmethod
    def should_crawl(url: str) -> bool:
        return url.startswith("https://samplesite.com/some-prefix-")
```

Usually, sitemaps are compound by more sitemaps, and this method will tell the crawler **which of those sub sitemaps are desired**. A common approach to check if this specific sitemap is useful or not is **checking its url**, as it might contain some specific patterns to imply some meaning.

With all this done, we already have a fully functional crawler, but **it's not yet available to the library** for some common operations, such as trigerring a crawling  for this new site.

## Fine grained url filtering
Maybe you want to perform some checking for each url that is to be scraped in the future, in order to do this, you can *optionally* implement the `should_scrape` method that will check if each url is a valid one.

```python
class SampleCrawler(BaseCrawler):
    
    # crawler name and start sitemap url...
    # should_crawl function...

    def should_scrape(url: str) -> bool:
        return url.startswith("https://samplesite.com") and len(url) > 42
```

# Adding a crawler

To add a crawler, all we need to do is to add its class to the `INSTALLED_CRAWLERS` list in the `c4v.scraper.settings` file:
```python
import sample_crawler

INSTALLED_CRAWLERS: List[Type[BaseCrawler]] = [
    <More crawlers>,
    sample_crawler.SampleCrawler
]
```
Now our new crawler will be available for common operations with scrapers, it will be recognized by the `microscope.Manager` object and it will be available for the CLI

# Adding irrelevant articles filtering
Sometimes we need to create a dataset with **non-relevant labeled data**, and label a lot of rows by hand can be a tiresome and time consuming task. In this case, you can specify the crawler that you want to crawl **only links that hold irrelevant information**.

To do this, you can provide a list `IRRELEVANT_URLS` with **regex patterns** for links that you already know that are ensured to be irrelevant for your use case.
```python
class SampleCrawler(BaseCrawler):
    # name and start sitemap link...
    IRRELEVANT_URLS = [
        ".*samplesite.com/irrelevant-section1/.*",
        ".*samplesite.com/irrelevant-section2/.*",
        # ...
    ]

    # should_crawl definition...
```