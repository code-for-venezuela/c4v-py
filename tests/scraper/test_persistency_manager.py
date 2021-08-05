from c4v.scraper.persistency_manager.sqlite_storage_manager import SqliteManager
from tests.scraper.utils import util_test_filter_scraped_urls, util_test_save_for, util_test_save_overrides_for


def test_save_sqlite_manager(test_sqlite_manager : SqliteManager):
    util_test_save_for(test_sqlite_manager)

def test_overrides_sqlite_manager(test_sqlite_manager : SqliteManager):
    util_test_save_overrides_for(test_sqlite_manager)

def test_list_sqlite_manager(test_sqlite_manager : SqliteManager):
    util_test_filter_scraped_urls(test_sqlite_manager)