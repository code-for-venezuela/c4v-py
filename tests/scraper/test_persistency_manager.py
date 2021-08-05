from c4v.scraper.persistency_manager.sqlite_storage_manager import SqliteManager
from tests.scraper.utils import util_test_filter_scraped_urls, util_test_instance_delete, util_test_save_for, util_test_save_overrides_for, util_test_url_filtering


def test_save_sqlite_manager(test_sqlite_manager : SqliteManager):
    """
        Test if save operation creates a new instance when that one does not exists
    """
    util_test_save_for(test_sqlite_manager)

def test_overrides_sqlite_manager(test_sqlite_manager : SqliteManager):
    """
        Test if save an ScrapedData instance overrides existent ones
    """
    util_test_save_overrides_for(test_sqlite_manager)

def test_list_sqlite_manager(test_sqlite_manager : SqliteManager):
    """
        Test listing of ScrapedData instances
    """
    util_test_filter_scraped_urls(test_sqlite_manager)

def test_filter_url_lists(test_sqlite_manager : SqliteManager):
    """
        Test filtering of ScrapedData instances
    """
    util_test_url_filtering(test_sqlite_manager)

def test_delete_row(test_sqlite_manager : SqliteManager):
    """
        Check that deletion works properly
    """
    util_test_instance_delete(test_sqlite_manager)