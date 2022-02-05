# Primary components
You can do everything the high level `miscroscope.Manager` api can by hand using its primary components instead, 
this way you can have better control if you need it.

## Scraper
The scraper component knows how to retrieve data from a known website. 

## Crawler
The crawler component is in charge of finding new urls for known sites.

## Persistency Manager
This component is in charge of managing how persistent data about articles is stored, as this might vary a lot depending on the kind of environment you need. You can provide a custom implementation for a persistency manager if you prefer. More on that [here](providing-a-custom-persistency-manager.md).

## Classifier
Use this component to infer data about a given article. The data we can gather right now is:

* Is this article a about issues with public services?
* If this article is about public services, which kind of service it is about?


