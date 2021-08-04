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
from c4v.scraper.scraper    import bulk_scrape
from c4v.scraper.settings   import INSTALLED_CRAWLERS

@click.group()
def c4v_cli():
    """
        Command entry point 
    """
    return

@c4v_cli.command()
@click.option('--files',    is_flag=True,   help="Interpret url list as files instead of urls, so urls are retrieved from such files. The file is expected to be formatted such that there's one url per line")
@click.option('--output',   default=None,   help="Output file to store results")
@click.option('--max-len',  default=-1,     help="Truncate scraped content's body to a maximum lenght if provided.")
@click.option('--pretty',   is_flag=True,   help="Print data formatted as human readable")
@click.argument('urls', nargs = -1)
def scrape(urls : List[str] = None, files : bool = None, output : str = None, max_len : int = -1, pretty : bool = False):
    """
        Use this command to run a scraping process.
        Parameters:
            + urls : [str] = List of urls or files to get urls from
            + files : bool = if urls are in files or not, defaults to false
            + output : str = where to store output 
            + max_len : int = max body size to print when pretty printing. Defaults to max size possible when not provided
            + pretty : bool = If print should be human readable
    """

    # Check for errors
    #   check if urls were provided
    if not urls: 
        click.echo("No urls to scrape were provided")
        return

    # Read urls
    urls_to_scrape = []
    if files: # if urls are stored in files
        for file in urls: # iterate over every file
            try:
                with open(file) as urls_file: 
                    content = map(lambda s: s.strip(), urls_file.readlines()) # parse every line as a single url
                    urls_to_scrape.extend(content)
            except IOError as e:
                click.echo(f"Could not open input file: {file}. Error: {e.strerror}")
    else:
        urls_to_scrape = urls

    # scrape urls 
    scraped_data = bulk_scrape(urls_to_scrape)

    # Choose print function 
    if pretty:
        print_func = lambda x: x.pretty_print(max_len)
    else:
        print_func = lambda x: str(x)

    # Write output
    if output:
        # try print content into file if output file was provided 
        try:
            with open(output, "w+") as file:
                for data in scraped_data:
                    print(print_func(data), file=file)
        except IOError as e:
            click.echo(f"Could not open {output} file. Error: {e.strerror}")
    else:
        for data in scraped_data:
            click.echo(print_func(data))



@c4v_cli.command()
@click.option('--list', is_flag=True, help="list available crawlers")
@click.option('--all', is_flag=True, help="Run all available crawlers")
@click.option('--all-but', is_flag=True, help="Run all crawlers except listed ones")
@click.option('--output', default=None, help="Store output in provided file")
@click.argument('crawlers', nargs=-1)
def crawl(crawlers : List[str] = [], list : bool = False, all : bool = False, all_but : bool = False, output : str = None):
    """
        Start a crawling process 
    """
    crawlable_sites = "".join([f"\t{crawl.name}\n" for crawl in INSTALLED_CRAWLERS])
    
    # list available crawlers if requested to 
    if list:
        click.echo(f"Crawlable sites: \n{crawlable_sites}")
        if not crawlers and not all: # if nothing else to do, just end
            return

    # Check for errors:
    # if no crawlers to use, just end
    if not crawlers and not all:
        click.echo(f"Not crawlers provided. Available crawlers:\n{crawlable_sites}")
        return

    # raise a warnning if incompatible flags where provided
    if all_but and all:
        click.echo(f"[WARNING] --all and --all-but incompatible flags were provided, using only all")

    # set up crawlers to run 
    if all:
        crawlers_to_run = INSTALLED_CRAWLERS
    elif all_but:
        crawlers_to_run = [crawler for crawler in INSTALLED_CRAWLERS if crawler.name not in crawlers]
    else:
        crawlers_to_run = [crawler for crawler in INSTALLED_CRAWLERS if crawler.name in crawlers]

    # function to format crawled urls list 
    format_url_list = lambda list: "".join([f"{s}\n" for s in list])

    # if output file provided, crawl 
    if output:
        try:
            with open(output, "w+") as file:
                # create process function
                def process(list : List[str]):
                    print(format_url_list(list), file=file)
                
                # crawl urls
                for crawler in crawlers_to_run:
                    c = crawler()
                    c.crawl_and_process_urls(process)

            return
        except IOError as e:
            click.echo(f"[ERROR] Unable to open file: {output}. Error: {e.strerror}")
            return

    # if output file not specified, just print to stdio
    for crawler in crawlers_to_run:
            c = crawler()

            def process(list : List[str]):
                click.echo(format_url_list(list))

            c.crawl_and_process_urls(process)
    

if __name__ == "__main__":
    c4v_cli()
