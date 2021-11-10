"""
    Entry point for the streamlit app
"""
import dataclasses
from typing import List
import streamlit as sl
import app

sl.write("# 🔬 C4V Microscope")
sl.write(
    "In the following dashboard we will see how the scraping, crawling and classification works."
)

app = app.App()

# -- < Filtering > ---------------------------------------------
# Add a title to the sidebar
sl.sidebar.write("## Filters")
sl.sidebar.write("Specify which data you want to see with the following filters")

# Add a selector to filter by label
label = sl.sidebar.selectbox(
    "Label: ", options=app.label_options, help="Label assigned by the classifier"
)
scraped = sl.sidebar.selectbox(
    "Scraped: ",
    options=app.scraped_options,
    help="Whether the instance data is complete or not",
)
use_max_rows = sl.sidebar.checkbox(
    "Limit Rows: ", value=True, help="Max ammount of rows to show"
)
# -- < Row Limits > -------------------------------------------
# If limiting the max amount of rows, then ask for a max value
if use_max_rows:
    max_rows = sl.sidebar.number_input(
        "Max: ", 0, value=100, help="Maximum amount of rows to retrieve and show"
    )
else:
    max_rows = -1

# -- < Classification controls > ------------------------------
# Help Message
sl.sidebar.write("-----")
sl.sidebar.write("## Classification controls")
sl.sidebar.write(
    "Run a classification process, select parameters with the following controls."
)

# Branch names
NO_BRANCH_NAME = "No Branch"
branch_name = sl.sidebar.selectbox(
    "Branch: ",
    [NO_BRANCH_NAME] + app.available_branchs,
    help="Experiment branch to select a classifier model from",
)
experiment_options = (
    []
    if branch_name == NO_BRANCH_NAME
    else app.available_experiments_for_branch(branch_name)
)
experiment_name = sl.sidebar.selectbox(
    "Experiment: ",
    experiment_options,
    help="Experiment name corresponding to provided branch",
)

# Description for experiment if provided
if experiment_name:
    sl.sidebar.write("### Summary for this experiment: ")
    sl.sidebar.text_area(
        "Summary", app.experiment_summary(branch_name, experiment_name), height=100
    )

# Set limits
use_classiffication_limit = sl.sidebar.checkbox(
    "Limit",
    value=True,
    help="Max ammount of rows to use during classification. Might be useful if you're running out of memory",
)
if use_classiffication_limit:
    max_rows_to_classify = sl.sidebar.number_input(
        "Max rows",
        0,
        value=100,
        help="Max amount of rows to classify. If you want no limit, uncheck the checkbox above",
    )
else:
    max_rows_to_classify = -1

# Run classification button
def run_classification_callback():
    if not experiment_name:
        sl.warning(
            "No experiment set up. Select one to choose a classifier model to use during classification"
        )
    else:
        sl.info("Running classification process, this may take a while...")
        try:
            app.classify(branch_name, experiment_name, max_rows_to_classify)
            sl.success("Classification finished")
        except Exception as e:
            sl.error(
                f"Unable to classify using model {branch_name}/{experiment_name} {('and up to ' + str(max_rows_to_classify)) if max_rows_to_classify >= 0 else ''}.    "
                + f"Error: {e}"
            )


sl.sidebar.button(
    "Classify",
    help="Perform the classification process",
    on_click=run_classification_callback,
)

# -- < Crawling Controls >  -----------------------------------
sl.sidebar.write("------")
sl.sidebar.write("## Crawling")
sl.sidebar.write("Crawl for new urls in the available sites")

# Limit
use_crawl_limit = sl.sidebar.checkbox(
    "Limit", help="Use a max amount of urls to be crawled at the same time", value=True
)
if use_crawl_limit:
    crawl_limit = sl.sidebar.number_input("Max instances to add", 0, value=100)
else:
    crawl_limit = -1
# Crawlers to use
crawlers = sl.sidebar.multiselect(
    "Crawl for: ",
    options=app.manager.get_available_crawlers(),
    help="Sites that can be scraped for new data",
)

# utility class to keep count of how much instance were crawled so far
@dataclasses.dataclass
class Count:
    count: int = 0

    def add(self, additional: int):
        self.count += additional


crawled_so_far = Count()

# Function to print a message with the current amount of crawled urls
def generate_progress_infinite():

    place_holder_msg = sl.empty()

    def progress_infinite(urls: List[str]):
        # Display some text telling how much urls have been crawled so far
        crawled_so_far.add(len(urls))
        place_holder_msg.info(f"Crawled {crawled_so_far.count} urls...")

    return progress_infinite


# Function to generate a finite progress bar function
def generate_progress_finite():
    sl.write("Progress")
    progress_bar = sl.progress(0)
    progress_msg = sl.empty()
    progress_msg.markdown(f"Crawled {crawled_so_far.count}/{crawl_limit}...")

    def progress_finite(urls: List[str]):
        crawled_so_far.add(len(urls))
        progress_msg.markdown(f"Crawled {crawled_so_far.count}/{crawl_limit}...")
        progress_bar.progress(int(100 * (crawled_so_far.count / crawl_limit)))

    return progress_finite


def run_crawl_callback():
    """
    Run a classification and show an info tect
    """
    # If nothing to do, just end the process
    if not crawlers or crawl_limit == 0:
        return

    # Start classification process
    sl.info("Starting crawling process")
    app.crawl(
        crawlers,
        crawl_limit,
        generate_progress_finite() if crawl_limit > 0 else generate_progress_infinite(),
    )

    # Tell the user how much urls were crawled
    if crawled_so_far.count == 0:
        sl.warning("No urls were crawled. Maybe there were no urls to crawl?")
        return

    sl.success(f"Succesfully crawled {crawled_so_far.count} elements!")


# Button to run
sl.sidebar.button(
    "Crawl",
    help="Run a crawling limit with the provided limits",
    on_click=run_crawl_callback,
)

# -- < Scraping controllers > ---------------------------------
sl.sidebar.write("------")
sl.sidebar.write("## Scraping")
sl.sidebar.write("Scrape instances pending for scraping")

# Limit
use_scrape_limit = sl.sidebar.checkbox(
    "Limit", help="Use a max amount of urls to be scraped at the same time", value=True
)
if use_scrape_limit:
    scrape_limit = sl.sidebar.number_input("Max instances to scrape", 0, value=100)
else:
    scrape_limit = -1

# Called when crawl button is pressed
def scrape_callback():
    sl.info("Starting scraping process...")
    if app.scrape(scrape_limit) == 0:
        sl.success("Successfull scraping! 🤩")
    else:
        sl.error("Scraping process crashed 😵")


# Scrape button
sl.sidebar.button(
    "Scrape",
    help="run a classification process, trying to scrape data for instances with incomplete data",
    on_click=scrape_callback,
)

# -- < Show Dashboard > ---------------------------------------
sl.dataframe(app.get_dashboard_data(label=label, max_rows=max_rows, scraped=scraped))
