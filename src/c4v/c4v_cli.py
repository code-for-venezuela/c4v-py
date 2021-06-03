"""
    For now, this is going to be the main CLI executable,
    so we can test things in the meanwhile
"""
# Third party imports
import click

# Python imports
from typing import DefaultDict, List
import os

# Local imports
from c4v.scraper.scraped_data_classes.scraped_data import ScrapedData
from c4v.scraper.scraper import bulk_scrape

@click.group()
def c4v_cli():
    """
        Command entry point 
    """
    return

@c4v_cli.command()
@click.option('--files', is_flag=True, help="Interpret url list as files instead of urls, so urls are retrieved from such files")
@click.option('--output', default=None, help="Output file to store results")
@click.option('--max-len', default=-1, help="Truncate scraped content's body to a maximum lenght if provided.")
@click.option('--pretty', is_flag=True, help="Print data formatted as human readable")
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

if __name__ == "__main__":
    c4v_cli()
