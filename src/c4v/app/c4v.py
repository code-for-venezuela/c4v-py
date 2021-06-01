"""
    For now, this is going to be the main CLI executable,
    so we can test things in the meanwhile
"""
# Third party imports
import click

# Python imports
from typing import List
# Local imports
import c4v.scraper

@click.group()
def c4v():
    """
        Command entry point 
    """
    return

@c4v.command()
@click.option('--files', is_flag=True, help="Interpret url list as files instead of urls, so urls are retrieved from such files")
@click.option('--output', default=None, help="Output file to store results")
@click.argument('urls', nargs = -1)
def scrape(urls : List[str] = None, files : bool = None, output : str = None):
    """
        Use this command to run a scraping process.
        Parameters:
            + urls : [str] = List of urls or files to get urls from
            + files : bool = if urls are in files or not, defaults to false
            + output : str = where to store output 
    """

    # Read urls
    urls_to_scrape = []
    if files: # if urls are stored in files
        for file in urls: # iterate over every file
            with open(file) as urls_file: 
                content = map(lambda s: s.strip(), urls_file.readlines()) # parse every line as a single url
                urls_to_scrape.extend(content)
    else:
        urls_to_scrape = urls

    # scrape urls 
    if output:
        pass        


if __name__ == "__main__":
    c4v()
