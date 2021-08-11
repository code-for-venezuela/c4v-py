"""
    For now, this is going to be the main CLI executable,
    so we can test things in the meanwhile
"""
# Third party imports
from datetime import datetime
import click

# Python imports
from typing         import List, Tuple
from urllib.error   import HTTPError
import os

# Local imports
from c4v.scraper.scraped_data_classes.scraped_data import ScrapedData
from c4v.scraper.settings       import INSTALLED_CRAWLERS
from c4v.scraper.persistency_manager.sqlite_storage_manager import SqliteManager
from c4v.scraper.utils          import data_list_to_table_str
from c4v.scraper.settings       import INSTALLED_CRAWLERS
from c4v.config                 import settings
from c4v.microscope.manager     import Manager


# Folder to search for local files
DEFAULT_FILES_FOLDER = settings.c4v_folder
DEFAULT_DB = settings.local_sqlite_db or os.path.join(settings.c4v_folder, settings.local_sqlite_db_name)


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
    client = CLIClient(Manager(db_manager), urls, files)
    # Read urls
    urls_to_scrape = []

    if not urls:
        urls_to_scrape = [d.url for d in db_manager.get_all(up_to, scraped=False)]
    elif files:  # if urls are stored in files
        urls_to_scrape = client.get_urls(urls)
    else:
        urls_to_scrape = urls

    # scrape urls
    scraped_data = client.get_data_for_urls(urls_to_scrape, should_scrape=True)

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
        crawlers_to_run = [crawler.name for crawler in INSTALLED_CRAWLERS]
    elif all_but:
        crawlers_to_run = [
            crawler.name for crawler in INSTALLED_CRAWLERS if crawler.name not in crawlers
        ]
    else:
        crawlers_to_run = [
            crawler.name for crawler in INSTALLED_CRAWLERS if crawler.name in crawlers
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

@c4v_cli.command()
@click.option("--no-scrape", is_flag=True, help="Don't scrape if url is not found in DB")
@click.option("--file", is_flag=True, help="Get urls of news to classify from a file")
@click.argument("inputs", nargs = -1)
def classify(inputs : List[str] = [], no_scrape : bool = False, file : bool = False):
    """
        Run a classification over a given url or from a file 
    """

    # Validate input:
    n_args = len(inputs) 
    if n_args < 2: # Get at the least 2 args, experiment as branch/experiment and some url
        click.echo("[ERROR] Should provide at the least 2 arguments, experiment and at the least 1 url")
        return

    manager = Manager.from_local_sqlite_db(DEFAULT_DB)
    client  = CLIClient(manager, inputs[1:], file)
    
    # validate branch and name
    parsed_branch_and_name = CLIClient.parse_branch_and_experiment_from(inputs[0])
    if parsed_branch_and_name == None:
        return
    else:
        branch, experiment = parsed_branch_and_name

    # Now get data for each url
    data = client.get_data_for_urls(should_scrape=not no_scrape)
    
    # Try to classify given data
    try:
        results = manager.run_classification_from_experiment(branch, experiment, data)
    except ValueError as e:
        click.echo(f"[ERROR] Could not classify provided data.\n\tError: {e}")
        return
    
    # Pretty print results:
    for (url, result) in results.items():
        click.echo(f"\t{url}")
        for (key, value) in result.items():
            click.echo(f"\t\t* {key} : {value}")
    

@c4v_cli.command()
@click.option("--no-scrape", is_flag=True, help="Don't scrape if data is not available locally")
@click.argument("url", nargs=1)
def show(url : str, no_scrape : bool = False):
    """
        Show the entire data for a given URL
    """
    # Create manager object
    manager = Manager.from_local_sqlite_db(DEFAULT_DB)
    client  = CLIClient(manager, [url]) 

    data = client.get_data_for_urls(should_scrape = not no_scrape)

    # Check if could retrieve desired element 
    if not data:
        click.echo("[EMPTY]")
        return
    else:
        element = data[0]

    # Pretty print element:
    line_len = os.get_terminal_size().columns
    click.echo("+" + ("=" * (line_len - 2))  + "+")
    click.echo(f"\tUrl        : {element.url}")
    click.echo(f"\tTitle      : {element.title}")
    click.echo(f"\tAuthor     : {element.author}")
    click.echo(f"\tDate       : {element.date}")
    click.echo(f"\tCategories : {', '.join(element.categories) if element.categories else '<No Category>'}")
    click.echo(f"\tScraped    : {datetime.strftime(element.last_scraped, settings.date_format)}")
    click.echo("=" * line_len)
    cleaned_content = "\n".join(
                                [
                                    s for s in 
                                    [s.strip() for s in element.content.splitlines()]
                                    if s != ""
                                ]
                            ) 
    click.echo(cleaned_content)
    click.echo("+" + ("=" * (line_len - 2))  + "+")


class CLIClient:
    """
        This class will manage common operations performed by the CLI tool
    """

    def __init__(self, manager : Manager, urls : List[str] = [], from_files : bool = False):
        self._manager = manager

        # If urls are in files, load them from such files
        if from_files:
            self._urls = CLIClient._parse_lines_from_files(urls)
        else:
            self._urls = urls
    
    def get_data_for_urls(self, urls : List[str] = None, should_scrape : bool = True) -> List[ScrapedData]:
        """
            Return a list of ScrapedData from a list of urls.
            Parameters:
                urls : [str] = List of urls whose data is requested. If not provided, defaults to the 
                                list stored whithin this class
                should_scrape : bool = If should scrape data for urls that are not currently available
            Return:
                List of retrieved urls
        """
        urls_to_retrieve = urls or self._urls

        # Check scrapable urls:
        scrapable_urls, non_scrapables = self._manager.split_non_scrapable(urls_to_retrieve)
        
        # Warn the user that some urls won't be scraped
        if non_scrapables:
            click.echo("[WARNING] some urls won't be retrieved, as they are not scrapable for now.", err=True)
            click.echo("Non-scrapable urls:", err=True)
            click.echo("\n".join([f"\t* {url}" for url in non_scrapables]), err=True)

        # check if some http error happens
        data = []
        try:
            data = self._manager.get_bulk_data_for(scrapable_urls, should_scrape = should_scrape)
        except HTTPError as e:
            click.echo(f"[ERROR] Could not scrape all data due to connection errors: {e}", err=True)
        
        # Tell the user if some urls where not retrieved
        succesfully_retrieved = {d.url for d in data}
        if any(url not in succesfully_retrieved for url in scrapable_urls):
            click.echo(f"[WARNING] Some urls couldn't be retrieved: ")
            click.echo("\n".join([f"\t* {url}" for url in scrapable_urls if url not in succesfully_retrieved]))

        return data
    
    def get_urls(self) -> List[str]:
        """
            Return stored urls
        """
        return self._urls

    @staticmethod
    def parse_branch_and_experiment_from(line : str) -> Tuple[str, str]:
        """
            Return a tuple with branch name and experiment name from a line representing 
            and experiment name. Report error if not a valid name
            Parameters:
                line : str = line to parse as branch name and experiment name. For example:
                        branch/exp1
                        branch-exp1
                    Invalid examples:
                        branch exp1 
                        branch\exp1
            Return:
                A tuple, with the first member as the branch name ans the second one as the experiment name
        """
        separators = ["/", "-"]
        for separator in separators:
            branch_and_name = line.split(separator)
            if len(branch_and_name) == 2:
                branch, name = branch_and_name
                return (branch, name)

        click.echo(f"[ERROR] Given experiment name is not valid: {line}. Should be in the form:", err=True)
        for separator in separators:
            click.echo(f"\tbranch_name{separator}experiment_name")
        return None
        
        

    
    @staticmethod
    def _parse_lines_from_files(files : List[str]) -> List[str]:
        """
            Utility function to collect all lines from multiple files
        """
        lines = []
        for file in files:  # iterate over every file
            try:
                with open(file) as file_with_lines:
                    content = map(
                        lambda s: s.strip(), file_with_lines.readlines()
                    )  # parse every line as a single url
                    lines.extend(content)
            except IOError as e:
                click.echo(f"Could not open input file: {file}. Error: {e.strerror}", err=True)
        return lines

if __name__ == "__main__":
    c4v_cli()
