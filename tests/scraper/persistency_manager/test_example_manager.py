from c4v.scraper.persistency_manager.example_dict_manager import DictManager # Replace here with your new class

# These are the functions you will use to to 
# test your new class. Of course, you can and should add
# more tests and test functions if your new class requires it.
# These are the bare minimum testing functions. 
from tests.scraper.utils import (
    util_test_filter_scraped_urls, 
    util_test_get_in_order, 
    util_test_instance_delete, 
    util_test_order_parsing, 
    util_test_save_for, 
    util_test_save_overrides_for, 
    util_test_url_filtering
    )


def test_save_example_manager(test_example_manager : DictManager):
    """
        Test if save operation creates a new instance when that one does not exists
    """
    util_test_save_for(test_example_manager)

def test_overrides_example_manager(test_example_manager : DictManager):
    """
        Test if save an ScrapedData instance overrides existent ones
    """
    util_test_save_overrides_for(test_example_manager)

def test_list_example_manager(test_example_manager : DictManager):
    """
        Test listing of ScrapedData instances
    """
    util_test_filter_scraped_urls(test_example_manager)

def test_filter_url_lists(test_example_manager : DictManager):
    """
        Test filtering of ScrapedData instances
    """
    util_test_url_filtering(test_example_manager)

def test_delete_row(test_example_manager : DictManager):
    """
        Check that deletion works properly
    """
    util_test_instance_delete(test_example_manager)

def test_get_in_order(test_example_manager : DictManager):
    """
        Check that ordering works properly in get_all function
    """
    util_test_get_in_order(test_example_manager)

def test_order_parsing(test_example_manager : DictManager):
    """
        Check that invalid formats for ordering are handled with ValueError Exceptions
    """
    util_test_order_parsing(test_example_manager)