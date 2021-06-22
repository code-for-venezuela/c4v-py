"""
    This file implements a class that manages local storage 
    using regular files.
"""
# third party imports
import json
import os
import sys

from typing import Any, Callable, Dict, List

# Local imports 
from c4v.scraper.persistency_manager.base_persistency_manager import BasePersistencyManager
from c4v.scraper.scraped_data_classes.scraped_data import ScrapedData, ScrapedDataEncoder

class JsonManager(BasePersistencyManager):
    """
        Use json based local storage 
        to manage persistency.

        File will have a format like:
        [ScrapedData]
    """

    def __init__(self, file : str, create_file_if_not_exist : bool = False):
        self._file = file

        # If file does not exist, create it
        if create_file_if_not_exist and not os.path.exists(file):
            with open(file, "w+"):
                pass

    def _load_json_file_content(self) -> List[ScrapedData]:
        """
            Read file to a dict
        """

        if os.stat(self._file).st_size == 0:
            with open(self._file, 'r+') as json_file:
                print(f"Warning: file  {self._file} is empty, filling it with empty list", file=sys.stderr)
                print("[]", file=json_file)


        with open(self._file, 'r') as json_file:
            return [_parse_dict_to_url_data(j) for j in json.load(json_file)]
        
    def _save_to_json(self, url_datas : List[ScrapedData]):
        """
            Save this list of ScrapedData to a local json file specified by 
            the private instance variable _file
        """
        with open(self._file, "w+") as json_file:
            json.dump(url_datas, json_file, cls=ScrapedDataEncoder)    

    def save(self, url_data: List[ScrapedData]):
        # Save given urls to disk. If there's an entry with 
        # an url matching with one of the instance variables, then save it 
        url_to_data = { url_d.url : url_d for url_d in self._load_json_file_content() }

        for url_d in url_data:
            url_to_data[url_d.url] = url_d

        # Save this file to local storage
        self._save_to_json(list(url_to_data.values()))
        

    def delete(self, urls: List[str]):
        # delete entries whose url is one of the listed ones
        urls_to_data = { url_d.url : url_d for url_d in  self._load_json_file_content()}

        for url in urls:
            if urls_to_data.get(url): del urls_to_data[url]
        
        self._save_to_json(urls_to_data.values())

    def get_matching(self, predicate: Callable[[ScrapedData], bool]) -> List[ScrapedData]:
        # Get instances matching this predicate
        return list(filter(predicate,self._load_json_file_content()))

    def filter_scraped_urls(self, urls: List[str]) -> List[str]:

        data = { d.url : d.last_scraped != None for d in self._load_json_file_content()}

        l = [
            url for url in urls if not data.get(url)
        ]
        # Get instances that should be scraped
        return l

    def was_scraped(self, url: str) -> bool:
        # Check if given url was scraped
        for url_d in self._load_json_file_content():
            if url_d.url == url:
                return url_d.last_scraped != None
        
        return False

def _parse_dict_to_url_data(obj : Dict[str, Any]) -> ScrapedData:
    """
    Parse dict object into a new object instance
    """
    return ScrapedData(**obj)

