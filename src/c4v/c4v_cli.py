"""
    For now, this is going to be the main CLI executable,
    so we can test things in the meanwhile
"""
# Third party imports
from datetime import datetime
import click

# Python imports
from typing import List, Tuple
from urllib.error import HTTPError
import os
from pathlib import Path

# Local imports
from c4v.scraper.scraped_data_classes.scraped_data import ScrapedData
from c4v.scraper.settings import INSTALLED_CRAWLERS
from c4v.scraper.persistency_manager.sqlite_storage_manager import SqliteManager
from c4v.scraper.utils import data_list_to_table_str
from c4v.scraper.settings import INSTALLED_CRAWLERS
from c4v.config import settings
from c4v.microscope.manager import Manager


# Folder to search for local files
DEFAULT_FILES_FOLDER = settings.c4v_folder
DEFAULT_DB = settings.local_sqlite_db or os.path.join(
    settings.c4v_folder, settings.local_sqlite_db_name
)


@click.group()
def c4v_cli():
    """
        Command entry point 
    """
    # init files if necessary:
    path = Path(DEFAULT_FILES_FOLDER)
    if not path.exists():
        click.echo(f"[INFO] Creating local files folder at: {DEFAULT_FILES_FOLDER}")
        try:
            path.mkdir(parents=True)
        except Exception as e:
            click.echo(f"[ERROR] Could not create '{path}' folder: {e}", err=True)
    elif not path.is_dir():
        click.echo(
            f"[ERROR] Files folder '{path}' already exists but it's not a file.",
            err=True,
        )


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
    "--limit",
    default=-1,
    help="Maximum amount of elements to scrape. Scrape all if no argument given",
)
@click.argument("urls", nargs=-1)
def scrape(
    urls: List[str] = None, files: bool = None, loud: bool = False, limit: int = -1
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

    manager = Manager.from_default()
    client = CLIClient(manager, urls, files)

    # Read urls
    urls_to_scrape = []

    if not urls:
        urls_to_scrape = [d.url for d in manager.get_all(limit, scraped=False)]
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
@click.option("--limit", default=-1, help="Max number of new urls to store")
@click.argument("crawlers", nargs=-1)
def crawl(
    crawlers: List[str] = [],
    list: bool = False,
    all: bool = False,
    all_but: bool = False,
    loud: bool = False,
    limit: int = -1,
):
    """
        Crawl for new urls, ignoring already scraped ones.\n
        Parameters:\n
            + list : bool = if should print list of available crawlers by name\n
            + all  : bool = if should crawl all crawlers. Not true by default as a crawling process may be really slow\n
            + all_but : bool = run all crawlers except the ones provided as arguments\n
            + loud : bool = if should print scraped urls to stdio\n
    """

    # Create default CLI client
    client = CLIClient()

    # list available crawlers if requested to
    crawlable_sites = "".join([f"\t{crawl.name}\n" for crawl in INSTALLED_CRAWLERS])
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
            f"[WARNING] --all and --all-but incompatible flags were provided, using only --all"
        )

    # set up crawlers to run
    if all:
        crawlers_to_run = [crawler.name for crawler in INSTALLED_CRAWLERS]
    elif all_but:
        crawlers_to_run = [
            crawler.name
            for crawler in INSTALLED_CRAWLERS
            if crawler.name not in crawlers
        ]
    else:
        crawlers_to_run = [
            crawler.name for crawler in INSTALLED_CRAWLERS if crawler.name in crawlers
        ]

    client.crawl_new_urls_for(crawlers_to_run, limit, loud)


@c4v_cli.command()
@click.option("--urls", is_flag=True, help="Only list urls")
@click.option("--limit", default=100, help='List only up to "limit" rows')
@click.option("--col-len", default=50, help="Columns max length")
@click.option("--count", is_flag=True, help="Print only count of selected data")
@click.option(
    "--scraped-only",
    default=None,
    help="Retrieve only complete rows, those with its scraped data",
)
def list(
    urls: bool = False,
    limit: int = 100,
    col_len: int = 50,
    count: bool = False,
    scraped_only: bool = None,
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
        (scraped_only == "true" or scraped_only == "True" or scraped_only == "1")
        if scraped_only
        else None
    )

    manager = Manager.from_default()

    # Just print urls if requested so
    if urls:
        for data in manager.get_all(limit):
            click.echo(data.url)
        return

    data = [d for d in manager.get_all(limit, scraped_only)]

    if count:
        click.echo(len(data))
        return

    # Get printable version of retrieved data
    data_to_print = data_list_to_table_str(data, max_cell_len=col_len)

    click.echo(data_to_print)


@c4v_cli.command()
@click.option(
    "--no-scrape", is_flag=True, help="Don't scrape if url is not found in DB"
)
@click.option("--file", is_flag=True, help="Get urls of news to classify from a file")
@click.option(
    "--limit",
    is_flag=False, 
    default=-1,
    help="Limit how much instances to classify in this run. Specially usefull when classifying pending data, if less than 0, then select as much as you can (default). Otherwise, classify at the most the given number",
    type=int,
)
@click.argument("inputs", nargs=-1)
def classify(
    inputs: List[str] = [], no_scrape: bool = False, file: bool = False, limit: int = -1
):
    """
        Run a classification over a given url or from a file, using the model stored in the provided
        experiment. Usage:
            c4v classify <branch_name>/<experiment_name> <url>
    """
    # Validate input:
    n_args = len(inputs)
    if (
        n_args < 2
    ):  # Get at the least 2 args, experiment as branch/experiment and some url
        click.echo(
            "[ERROR] Should provide at the least 2 arguments, experiment and at the least 1 url"
        )
        return

    # Create manager object
    manager = Manager.from_default()
    client = CLIClient(manager, file)

    # validate branch and name
    parsed_branch_and_name = CLIClient.parse_branch_and_experiment_from(inputs[0])
    if parsed_branch_and_name == None:
        return
    else:
        branch, experiment = parsed_branch_and_name

    # check if we have to classify pending data
    classify_pending = n_args == 2 and inputs[1] == "pending"
    if classify_pending:
        res = manager.run_pending_classification_from_experiment(
            branch, experiment, save=True, limit=limit
        )
        click.echo(f"[INFO] {len(res)} classified rows")
        return

    data = client.get_data_for_urls(urls=inputs[1:], should_scrape=not no_scrape)

    # Do nothing if not necessary:
    if not data:
        click.echo("[INFO] Nothing to classify")
        return

    # Try to classify given data
    try:
        results = manager.run_classification_from_experiment(branch, experiment, data)
    except ValueError as e:
        click.echo(f"[ERROR] Could not classify provided data.\n\tError: {e}", err=True)
        return
    except ModuleNotFoundError as e:
        click.echo(f"[ERROR] Could not found some modules, maybe you should try to change the installation profile of c4v. Erro: {e}", err=True)
        return

    # Pretty print results:
    for result in results:
        click.echo("\n")
        data: ScrapedData = result["data"]
        scores = result["scores"]
        click.echo(f"\t{data.title if data.title else '<no title>'} ({data.url})")
        click.echo(f"\t\t{data.label_relevance}")
        click.echo(f"\t\t{scores}")


@c4v_cli.command()
@click.option(
    "--no-scrape", is_flag=True, help="Don't scrape if data is not available locally"
)
@click.argument("url", nargs=1)
def show(url: str, no_scrape: bool = False):
    """
        Show the entire data for a given URL
    """
    # Create manager object
    manager = Manager.from_default()
    client = CLIClient(manager, [url])

    data = client.get_data_for_urls(should_scrape=not no_scrape)

    # Check if could retrieve desired element
    if not data:
        click.echo("[EMPTY]")
        return
    else:
        element = data[0]

    # Pretty print element:
    line_len = os.get_terminal_size().columns
    click.echo("+" + ("=" * (line_len - 2)) + "+")
    click.echo(f"\tUrl        : {element.url}")
    click.echo(f"\tTitle      : {element.title}")
    click.echo(f"\tAuthor     : {element.author}")
    click.echo(f"\tDate       : {element.date}")
    click.echo(
        f"\tCategories : {', '.join(element.categories) if element.categories else '<No Category>'}"
    )
    click.echo(
        f"\tScraped    : {datetime.strftime(element.last_scraped, settings.date_format)}"
    )
    click.echo("=" * line_len)
    cleaned_content = "\n".join(
        [s for s in [s.strip() for s in element.content.splitlines()] if s != ""]
    )
    click.echo(cleaned_content)
    click.echo("+" + ("=" * (line_len - 2)) + "+")


@c4v_cli.command()
def dashboard():
    """
        Launch the built in streamlit dashboard within the package might not be 
        available depending on your currently installed profile
    """
    # Try to import streamlit
    try:
        import streamlit 
        import streamlit.cli as slcli
    except ModuleNotFoundError as e:
        click.echo(f"[ERROR] Streamlit package not found, you might want to install the corresponding profile for this library. Erro {e}") 
        return
    
    import sys

    sys.argv = ["streamlit", "run", str(Path(Path(__file__).parent, "dashboard", "main.py"))]
    click.echo("[INFO] Starting c4v dashboard...")
    sys.exit(slcli.main())


@c4v_cli.command()
@click.option("--url", is_flag=True, help="Interpret string as URL")
@click.option(
    "--no-scrape",
    is_flag=True,
    help="In case of retrieving data from URL, don't scrape it if not available",
)
@click.option(
    "--html",
    default="./explanation.html",
    help="Dump results in an human readable format to an html file",
    nargs=1,
)
@click.option(
    "--label",
    default=None,
    help="Label to include in expalantion. If the predicted label is different from this one, then explain how much this label was contributing to its corresponding value",
)
@click.argument("experiment", nargs=1)
@click.argument("sentence", nargs=1)
def explain(
    experiment: str,
    sentence: str,
    url: bool = False,
    no_scrape: bool = False,
    html: str = None,
    label: str = None,
):
    """
        Show explainability for the given string. That is, show how much each word contributes 
        to each tag in the classifier. The result depends on the experiment, as it will load that 
        model to explain. The experiment argument follows the <branch_name>/<experiment_name> syntax
        Arguments:
            experiment : str = experiment format, following <branch_name>/<experiment_name> format
            sentence   : str = expression to explain 
    """
    microscope_manager = Manager.from_default()
    client = CLIClient(microscope_manager)

    # Get text to explain
    if url:
        # Get data to classify
        datas = client.get_data_for_urls([sentence], not no_scrape)

        # check if there's data retrieved to explain
        if not datas:
            click.echo("Nothing to explain")
            return

        # Get content from data retrieved
        text_to_explain = datas[0].content
    else:
        text_to_explain = sentence

    # Parse branch name and experiment
    branch_and_experiment = client.parse_branch_and_experiment_from(experiment)
    if branch_and_experiment == None:
        return

    # unpack branch name and experiment name
    branch, experiment = branch_and_experiment

    # Check label input
    try:
        possible_labels = microscope_manager.get_classifier_labels()
    except ModuleNotFoundError as e:
        click.echo(f"[ERROR] Could not found some modules, maybe you should try to change the installation profile of c4v. Erro: {e}", err=True)
        return

    if label and label not in possible_labels:
        click.echo(
            f"[WARNING] Provided label not a valid label, ignoring label argument {label}.",
            err=True,
        )
        click.echo(f"Possible Labels:", err=True)
        for l in possible_labels:
            click.echo(f"\t* {l}", err=True)
        label = None

    # try to explain
    try:
        explanation = microscope_manager.explain_for_experiment(
            branch, experiment, text_to_explain, html_file=html, additional_label=label
        )
    except ValueError as e:
        click.echo(f"[ERROR] Could not explain given sentence. Error: {e}")
        return

    # Pretty print results
    scores = explanation["scores"]
    label = explanation["label"]
    click.echo(f"Predicted Label: {label}\nScores:")
    for (word, score) in scores:
        click.echo(f"\t* {word} : {score}")


@c4v_cli.group()
def experiment():
    """
        Experiment Management. You can get info about experiments with this command, such as 
        listing, and removing them if no longer necessary
    """
    path = Path(settings.experiments_dir)
    if not path.exists():
        click.echo(f"[INFO] Creating experiments folder in: {path}")
        try:
            path.mkdir()
        except Exception as e:
            click.echo(
                f"[ERROR] Could not create folder due to the following error: {e}",
                err=True,
            )

    elif not path.is_dir():
        click.echo(
            f"[ERROR] Could not create folder {path}. File already exists but is not a folder",
            err=True,
        )


@experiment.command()
@click.argument("branch", nargs=1, required=False)
def ls(branch: str = None):
    """
        List branches if no argument is provided. If a branch name is specified, then list experiments within that branch.
        Examples:
            c4v experiment ls
                branch1
                branch2
                branch3
            c4v experiment ls branch1
                experiment1
                experiment2
    """
    # TODO tal vez mover esta lógica al ExperimentFSManager?
    if not branch:
        click.echo(f"[INFO] Listing from {settings.experiments_dir}")
        files = CLIClient._ls_files(settings.experiments_dir)
        click.echo("\n".join(files))
        return

    # Check if branch exists
    path = Path(settings.experiments_dir, branch)
    if not path.exists():
        click.echo(
            f"[ERROR] This is not a valid branch name: {branch} in {path}. You can see available branches using the command:\n\tc4v experiment ls",
            err=True,
        )
        return
    elif not path.is_dir():
        click.echo(
            f"[ERROR] Invalid Branch path: {path}. The branch name '{branch}' does not refers to an actual branch's directory"
        )

    # As everything is ok, just list files
    click.echo(f"[INFO] Listing from {path}")
    files = CLIClient._ls_files(path=str(path))
    click.echo("\n".join(files))


@experiment.command()
@click.argument("experiment", nargs=1)
def summary(experiment: str):
    """
        Print summary for an existent experiment given its name with branch and experiment name\n
        Example:\n
            `c4v experiment summary branch_name/experiment_name`
    """

    # Parse branch and experiment name
    branch_and_experiment = CLIClient.parse_branch_and_experiment_from(experiment)
    if not branch_and_experiment:
        return

    # as everything went ok, parse branch name and experimet
    branch_name, experiment_name = branch_and_experiment

    path = Path(settings.experiments_dir, branch_name, experiment_name, "summary.txt")
    # Check if file exists
    if not path.exists():
        click.echo(
            f"[ERROR] Summary for experiment {experiment} not found in {path}", err=True
        )
        return
    elif not path.is_file():
        click.echo(
            f"[ERROR] Sumamry for experiment {experiment} in {path} is not a valid file",
            err=True,
        )
        return

    # As everything went ok, print file content
    click.echo(f"[INFO] Reading summary from: {path}")
    click.echo(path.read_text())


class CLIClient:
    """
        This class will manage common operations performed by the CLI tool
    """

    def __init__(
        self, manager: Manager = None, urls: List[str] = [], from_files: bool = False
    ):

        # Default manager
        if not manager:
            manager = Manager.from_default(local_files_path=settings.c4v_folder)

        self._manager = manager

        # If urls are in files, load them from such files
        if from_files:
            self._urls = CLIClient._parse_lines_from_files(urls)
        else:
            self._urls = urls

    def get_data_for_urls(
        self, urls: List[str] = None, should_scrape: bool = True
    ) -> List[ScrapedData]:
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

        scrapable_urls, non_scrapables = self._manager.split_non_scrapable(
            urls_to_retrieve
        )

        # Warn the user that some urls won't be scraped
        if non_scrapables:
            click.echo(
                "[WARNING] some urls won't be retrieved, as they are not scrapable for now.",
                err=True,
            )
            click.echo("Non-scrapable urls:", err=True)
            click.echo("\n".join([f"\t* {url}" for url in non_scrapables]), err=True)

        # check if some http error happens
        data = []
        try:
            data = self._manager.get_bulk_data_for(
                scrapable_urls, should_scrape=should_scrape
            )
        except HTTPError as e:
            click.echo(
                f"[ERROR] Could not scrape all data due to connection errors: {e}",
                err=True,
            )

        # Tell the user if some urls where not retrieved
        succesfully_retrieved = {d.url for d in data}
        if any(url not in succesfully_retrieved for url in scrapable_urls):
            click.echo(f"[WARNING] Some urls couldn't be retrieved: ")
            click.echo(
                "\n".join(
                    [
                        f"\t* {url}"
                        for url in scrapable_urls
                        if url not in succesfully_retrieved
                    ]
                )
            )

        return data

    def get_urls(self) -> List[str]:
        """
            Return stored urls
        """
        return self._urls

    @staticmethod
    def parse_branch_and_experiment_from(line: str) -> Tuple[str, str]:
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

        click.echo(
            f"[ERROR] Given experiment name is not valid: {line}. Should be of the form:",
            err=True,
        )
        for separator in separators:
            click.echo(f"\tbranch_name{separator}experiment_name")
        return None

    def crawl_new_urls_for(
        self, crawler_names: List[str], limit: int = -1, loud: bool = False
    ):
        """
            Crawl URLs for the given crawlers, up to a max number of urls.
            Parameters:
                crawlers : [str] = List of crawlers names to use
                limit    : int = Max limit of urls to get. 
                loud     : bool = If should print to terminal obtained scrapers. False by default
        """
        # function to format crawled urls list
        format_url_list = lambda list: "".join([f"{s}\n" for s in list])

        def process(list: List[str]):
            if loud:
                click.echo(format_url_list(list))

        # Names for installed crawlers
        crawlers = [c.name for c in INSTALLED_CRAWLERS]

        not_registered = [name for name in crawler_names if name not in crawlers]

        # Report warning if there's some non registered crawlers
        if not_registered:
            click.echo(
                "WARNING: some names in given name list don't correspond to any registered crawler.",
                err=True,
            )
            click.echo(
                "Unregistered crawler names: \n"
                + "\n".join([f"\t* {name}" for name in not_registered]),
                err=True,
            )

        self._manager.crawl_and_process_new_urls_for(
            [c for c in crawler_names if c not in not_registered], process, limit=limit
        )

    @staticmethod
    def _parse_lines_from_files(files: List[str]) -> List[str]:
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
                click.echo(
                    f"Could not open input file: {file}. Error: {e.strerror}", err=True
                )
        return lines

    @staticmethod
    def _ls_files(path: str) -> List[str]:
        """
            Returns a list of file names within a given directory 
        """
        return [str(x.name) for x in Path(path).glob("*")]


if __name__ == "__main__":
    c4v_cli()
