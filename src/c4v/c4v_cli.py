"""
    For now, this is going to be the main CLI executable,
    so we can test things in the meanwhile
"""
# Third party imports
import click

# Python imports
from typing import List
import os

# Local imports
from c4v.scraper.scraper import bulk_scrape
from c4v.scraper.scraped_data_classes.scraped_data import ScrapedData
from c4v.scraper.settings import INSTALLED_CRAWLERS
from c4v.scraper.persistency_manager.sqlite_storage_manager import SqliteManager
from c4v.scraper.utils import data_list_to_table_str
from c4v.scraper.scraper    import bulk_scrape
from c4v.scraper.settings   import INSTALLED_CRAWLERS

# Folder to search for local files @TODO debo cambiar esto para escoger correctamente el sitio para el directorio
DEFAULT_FILES_FOLDER = os.environ.get("HOME") + "/.c4v"
DEFAULT_DB = os.path.join(DEFAULT_FILES_FOLDER, "c4v_db.sqlite")


@click.group()
def c4v_cli():
    """
        Command entry point 
    """
    # init files if necessary:
    if not os.path.isdir(DEFAULT_FILES_FOLDER):
        click.echo(
            f"[INFO] Creating local files folder at: {DEFAULT_FILES_FOLDER}", err=True
        )
        try:
            os.mkdir(DEFAULT_FILES_FOLDER)
        except Exception as e:
            print(e)


@c4v_cli.command()
@click.option(
    "--files",
    is_flag=True,
    help="Interpret url list as files instead of urls, so urls are retrieved from such files. The file is expected to be formatted such that there's one url per line",
)
@click.option(
    "--loud", is_flag=True, help="Print scraped data to stdio as it is being scraped"
)
@click.option(
    "--up-to",
    default=-1,
    help="Maximum amount of elements to scrape. Scrape all if no argument given",
)
@click.argument("urls", nargs=-1)
def scrape(
    urls: List[str] = None, files: bool = None, loud: bool = False, up_to: int = -1
):
    """
        Use this command to run a scraping process. Scraped urls are retrieved from database by default.\n
        If provided with a list of urls, then those urls are the ones to be scraped and stored. \n
        You can pass the --files flag to interpret those paths as paths to files storing the actual urls,\n
        formated as one url per line.\n
        Parameters:\n
            + urls : [str] = List of urls or files to get urls from\n
            + files : bool = if urls are in files or not, defaults to false\n
            + up_to : int = maximum amount of urls to scrape in this pass. Scrape all non-scraped if not provided\n
            + loud : bool = if should print scraped data once a scraping is finished\n
    """

    db_manager = SqliteManager(DEFAULT_DB)

    # Read urls
    urls_to_scrape = []

    if not urls:
        urls_to_scrape = [d.url for d in db_manager.get_all(up_to, scraped=False)]
    elif files:  # if urls are stored in files
        for file in urls:  # iterate over every file
            try:
                with open(file) as urls_file:
                    content = map(
                        lambda s: s.strip(), urls_file.readlines()
                    )  # parse every line as a single url
                    urls_to_scrape.extend(content)
            except IOError as e:
                click.echo(f"Could not open input file: {file}. Error: {e.strerror}")
    else:
        urls_to_scrape = urls

    # scrape urls
    scraped_data = bulk_scrape(urls_to_scrape)

    # Write output
    db_manager.save(scraped_data)

    # Print obtained results if requested
    if loud:
        click.echo(data_list_to_table_str(scraped_data))


@c4v_cli.command()
@click.option("--list", is_flag=True, help="list available crawlers")
@click.option("--all", is_flag=True, help="Run all available crawlers")
@click.option("--all-but", is_flag=True, help="Run all crawlers except listed ones")
@click.option("--loud", is_flag=True, help="Print results to terminal")
@click.argument("crawlers", nargs=-1)
def crawl(
    crawlers: List[str] = [],
    list: bool = False,
    all: bool = False,
    all_but: bool = False,
    loud: bool = False,
):
    """
        Crawl for new urls, ignoring already scraped ones.\n
        Parameters:\n
            + list : bool = if should print list of available crawlers by name\n
            + all  : bool = if should crawl all crawlers. Not true by default as a crawling process may be really slow\n
            + all_but : bool = run all crawlers except the ones provided as arguments\n
            + loud : bool = if should print scraped urls to stdio\n
    """
    crawlable_sites = "".join([f"\t{crawl.name}\n" for crawl in INSTALLED_CRAWLERS])

    # Create default db manager
    db_manager = SqliteManager(DEFAULT_DB)

    # list available crawlers if requested to
    if list:
        click.echo(f"Crawlable sites: \n{crawlable_sites}")
        if not crawlers and not all:  # if nothing else to do, just end
            return

    # Check for errors:
    # if no crawlers to use, just end
    if not crawlers and not all:
        click.echo(f"Not crawlers provided. Available crawlers:\n{crawlable_sites}")
        return

    # raise a warnning if incompatible flags where provided
    if all_but and all:
        click.echo(
            f"[WARNING] --all and --all-but incompatible flags were provided, using only all"
        )

    # set up crawlers to run
    if all:
        crawlers_to_run = INSTALLED_CRAWLERS
    elif all_but:
        crawlers_to_run = [
            crawler for crawler in INSTALLED_CRAWLERS if crawler.name not in crawlers
        ]
    else:
        crawlers_to_run = [
            crawler for crawler in INSTALLED_CRAWLERS if crawler.name in crawlers
        ]

    # function to format crawled urls list
    format_url_list = lambda list: "".join([f"{s}\n" for s in list])

    # if loud flag is provided, print it tu stdio
    for crawler in crawlers_to_run:
        c = crawler()

        def process(list: List[str]):
            scraped_data = [
                ScrapedData(url=url) for url in db_manager.filter_scraped_urls(list)
            ]
            db_manager.save(scraped_data)

            if loud:
                click.echo(format_url_list(list))

        c.crawl_and_process_urls(process)


@c4v_cli.command()
@click.option("--urls", is_flag=True, help="Only list urls")
@click.option("--limit", default=100, help='List only up to "limit" rows')
@click.option("--col-len", default=50, help="Columns max length")
@click.option(
    "--scraped-only",
    default=None,
    help="Retrieve only complete rows, those with its scraped data",
)
def list(
    urls: bool = False, limit: int = 100, col_len: int = 50, scraped_only: bool = None
):
    """
    List requested info as specified by arguments.\n
    Parameters:\n
        + urls : bool = list urls only\n
        + limit : int = max num of rows to show. Set -1 to list all\n
        + col_len : int = Maximum lenght of shown columns\n
        + scraped_only : bool = if true, retrieve already scraped ones only, if false, retrieve non scraped only. If not provided, return anything\n
    """

    scraped_only = (
        scraped_only == "true" or scraped_only == "True" or scraped_only == "1"
    )

    db_manager = SqliteManager(DEFAULT_DB)

    # Just print urls if requested so
    if urls:
        for data in db_manager.get_all(limit):
            click.echo(data.url)
        return

    # Get printable version of retrieved data
    data_to_print = data_list_to_table_str(
        [d for d in db_manager.get_all(limit, scraped_only)], max_cell_len=col_len
    )

    click.echo(data_to_print)
    print(scraped_only)

if __name__ == "__main__":
    c4v_cli()
