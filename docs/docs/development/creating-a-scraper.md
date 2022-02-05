# Creating a Scraper
A scraper is a component that receives some urls and returns all the data it can gather from that web page. Usually, **the kind of data you can get for each page might vary a lot**, but as we need to store data for a lot of sites, we need to make this data canonical for every site. In order to do this, data is usually scraped into a specific format for each site, and then such format should **provide a way to transform itself into the canonical data format**, the `ScrapedData` dataclass.

!!!Info
    More about python dataclasses [here](https://docs.python.org/3/library/dataclasses.html)

The easiest way to create a new scraper is implement a *scrapy-based scraper*, the following section explains how to achieve such thing. If you need a fine-grained approach to scrape a specific web page, that's possible as well, by implementing the `BaseScraper` class. 

## Creating a Scrapy-based scraper
To create a Scrapy-based scrapers, we have to follow three simple steps:

1. **Create the data format**: We will create a data format describing the data we can gather from our site.
2. **Create a spider**: A scrapy object we use to parse data from the site
3. **Create a scraper class**: The object that describes a scraper to the `c4v` library
4. **Wire the scraper to the library**: Connect your brand new scraper to the rest of the features that the `c4v` library has to offer.

### Creating the data format
Let's say that we want to scrape articles for the site `samplesite.com`. First, we need to **create a data format for our specific site**, which will describe the data we know we can extract for every article in such site. To do so, we create a `dataclass` inheriting the base class `BaseDataFormat` in `c4v.scraper.scraped_data_classes.base_scraped_data`:
```python
from dataclasses import dataclass
from c4v.scraper.scraped_data_classes.base_scraped_data import BaseDataFormat

@dataclass(frozen=True)
class SampleSiteData(BaseDataFormat):
    """
        Base data format for samplesite.com
    """
    tags: List[str]
    categories: List[str]
    title: str
    author: str
    date: str
    body: str

```

Now that we have our data format, we have to impement the `to_scraped_data` method, so the scraper knows how to map from this data to the canonical `ScrapedData` format:

```python
from dataclasses import dataclass
from c4v.scraper.scraped_data_classes.base_scraped_data import BaseDataFormat

@dataclass(frozen=True)
class SampleSiteData(BaseDataFormat):
    # Fields...

    def to_scraped_data(self) -> ScrapedData:
        return ScrapedData(
            author=self.author,
            date=self.date,
            title=self.title,
            categories=self.tags + self.categories,
            content=self.body,
            url=self.url,
            last_scraped=self.last_scraped,
            source=Sources.SCRAPING,
        )
```

!!!Warning
    The `source` field is important as it tells you where this `ScrapedData` instance came from. In this case, it came from a scraper.

### Creating the spider
Now that we have our data format,  we need to create a scrapy spider that will retrieve information for each article. The process for creating a scrapy spider is well documented in the [scrapy documentation](https://docs.scrapy.org/en/latest/intro/tutorial.html#our-first-spider), so the only thing you need to know is that the `parse` method for your spider should return an instance of the data format we just created, the `SampleSiteData` class.

### Creating the scraper class
Now we need to create a scraper class that inherits the `BaseScrapyScraper` class. This class will describe our scraper to the rest of the library:

```python
from c4v.scraper.scrapers.base_scrapy_scraper import BaseScrapyScraper

class SampleSiteScraper(BaseScrapyScraper):
    """
        Scrapes data from SampleSite
    """

    intended_domain = "samplesite.com"
    spider = SampleSiteSpider

```

1. With `intender_domain` we're telling the library which urls are scrapable with this class (the ones that belong to this domain)
2. With `spider`, we're telling the scraper that this is the spider it should use when scraping the site with scrapy

### Wiring the scraper
Now that we have a fully functional scraper, we might want to add it to the `c4v` library as another scraper, so it can be used in the cli tool, the dashboard, and the `microscope.Manager` object.

To do this, the only thing we have to do is go to the `c4v.scraper.settings` module and add our scraper to the `INSTALLED_SCRAPERS` list:
```python
INSTALLED_SCRAPERS: List[Type[BaseScraper]] =
    [
        # More scrapers, 
        SampleSiteScraper
    ]
```

And that's it, now you have a fully functional scraper that it's available to the entire `c4v` library 🎉.
