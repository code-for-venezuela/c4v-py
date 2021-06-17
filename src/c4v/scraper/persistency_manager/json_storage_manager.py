"""
    This file implements a class that manages local storage 
    using regular files.
"""
# third party imports
import dataclasses
import json
from json import encoder
import os

from typing import Any, Callable, Dict, List
from typing_extensions import Required

# Local imports 
from c4v.scraper.persistency_manager.base_persistency_manager import BasePersistencyManager
from c4v.scraper.crawler.url_data import UrlData, UrlDataEncoder
from c4v.scraper.scraped_data_classes.scraped_data import ScrapedData

class JsonManager(BasePersistencyManager):
    """
        Use json based local storage 
        to manage persistency.

        File will have a format like:
        [UrlData]
    """

    def __init__(self, file : str, create_file_if_not_exist : bool = False):

        self._file = file

        # If file does not exist, create it
        if create_file_if_not_exist and not os.path.exists(file):
            with open(file, "w+"):
                pass
        
        # Load file content to class
        self._load_json_file_content()

    def _load_json_file_content(self) -> List[UrlData]:
        """
            Read file to a dict
        """
        with open(self._file, 'r') as json_file:
            return json.load(json_file, object_hook=_parse_file_data_to_url_data)
        
    def _save_to_json(self, url_datas : List[UrlData]):
        """
            Save this list of UrlData to a local json file specified by 
            the private instance variable _file
        """
        with open(self._file, "w+") as json_file:
            json.dump(url_datas, json_file, cls=UrlDataEncoder)    

    def save(self, url_data: List[UrlData]):

        url_to_data = { url_d.url : url_d for url_d in self._load_json_file_content() }

        for url_d in url_data:
            url_to_data[url_d.url] = url_d

        # Save this file to local storage
        self._save_to_json(url_to_data.values())
        

    def delete(self, urls: List[str]):

        urls_to_data = { url_d.url : url_d for url_d in  self._load_json_file_content()}

        for url in urls:
            if urls_to_data.get(url): del urls_to_data[url]
        
        self._save_to_json(urls_to_data.values())

    def get_matching(self, predicate: Callable[[UrlData], bool]) -> List[UrlData]:
        return list(filter(predicate,self._load_json_file_content()))

    def filter_scraped_urls(self, urls: List[str]) -> List[str]:
        return self.get_matching(lambda url_d : url_d.last_scraped == None)

    def was_scraped(self, url: str) -> bool:
        for url_d in self._load_json_file_content():
            if url_d.url == url:
                return url_d.last_scraped == None

def _parse_file_data_to_url_data(obj : Dict[str, Any]) -> UrlData:
    """
    Parse dict object into a new object instance
    """
    scraped_data_dict = obj['scraped_data']

    scraped_data = None
    if scraped_data_dict :
        scraped_data = ScrapedData( **scraped_data_dict  )


    return UrlData(
            url=obj['url'], 
            last_scraped=obj['last_scraped'], 
            scraped_data=scraped_data
        )

