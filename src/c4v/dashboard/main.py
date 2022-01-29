"""
    Entry point for the streamlit app
"""

# Python imports
import dataclasses
from typing import List
from pathlib import Path

# Third party imports
import streamlit as sl

# Internal imports
import c4v.dashboard.app as app
from c4v.config import settings

sl.write("# 🔬 C4V Microscope")
sl.write(
    "In the following dashboard we will see how the scraping, crawling and classification works."
)

# -- < Back End sub menu > -------------------------------------
backend = sl.sidebar.write("## Select Backend")

sl.sidebar.write(
    """
        Select a backend. If using a local backend, operations like scraping and crawling
        will be performed in you're local computer, and displayed data will come from
        your local sqlite database. 

        If you select a cloud backend, you'll need to set up the appropiate permissions
        and configuration environment variables, and operations will be performed
        on cloud. Displayed data will come from the cloud as well.
    """
)

backend = sl.sidebar.radio("Backend", ["local", "cloud"])
if backend == "local":
    app = app.App()
elif backend == "cloud":
    app = app.CloudApp()
else:
    raise ValueError(f"Invalid backend option: '{backend}'")

# -- < Filtering > ---------------------------------------------
# Add a title to the sidebar
sl.sidebar.write("## Filters")
sl.sidebar.write("Specify which data you want to see with the following filters")

# Add a selector to filter by label
relevance = sl.sidebar.selectbox(
    "Relevance: ",
    options=app.relevance_options,
    help="relevance label assigned by the classifier",
)

service = sl.sidebar.selectbox(
    "Service: ",
    options=app.service_options,
    help="service label assigned by the classifier",
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
sl.sidebar.write("## Classification")
sl.sidebar.write(
    "Run a classification process, select parameters with the following controls."
)

# Classification type
classifier_type = sl.sidebar.selectbox(
    "Type: ",
    ["relevance", "service"],
    help="The classifier type specifies which information is predicted by the classifier",
)

# Branch names
NO_BRANCH_NAME = "No Branch"
if backend == "local":
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
def run_local_classification_callback():
    if not experiment_name:
        sl.warning(
            "No experiment set up. Select one to choose a classifier model to use during classification"
        )
    else:
        sl.info("Running classification process, this might take a while...")
        try:
            app.classify(
                branch_name, experiment_name, max_rows_to_classify, type=classifier_type
            )
            sl.success("Classification finished")
        except MemoryError as e:
            sl.error(
                f"Unable to classify using model {branch_name}/{experiment_name} {('and up to ' + str(max_rows_to_classify)) if max_rows_to_classify >= 0 else ''}.    "
                + f"Error: {e}"
            )


def run_cloud_classification_callback():
    sl.info("Running classification process, this might take a while...")
    try:
        app.classify(classifier_type, max_rows_to_classify, type=classifier_type)
        sl.success("Classification finished")
    except Exception as e:
        sl.error(
            f"Unable to classify using model of type {classifier_type} {('and up to ' + str(max_rows_to_classify)) if max_rows_to_classify >= 0 else ''}.    "
            + f"Error: {e}"
        )


sl.sidebar.button(
    "Classify",
    help="Perform the classification process",
    on_click=run_local_classification_callback
    if backend == "local"
    else run_cloud_classification_callback,
)

# Downloading and uploading
def upload_and_download_model():
    sl.sidebar.write("### Uploading and downloading")
    sl.sidebar.write(
        """
                Upload a model to be used online during classification. 
                Specify the type of model to tell the purpose of such model
            """
    )
    # Check that a valid branch is provided
    if not branch_name and not experiment_name:
        sl.sidebar.write("⚠️ Provide branch name and experiment name to upload a model")
        return
    elif not branch_name:
        sl.sidebar.write("⚠️ Provide branch name to upload a model")
        return
    elif not experiment_name:
        sl.sidebar.write("⚠️ Provide experiment name to upload a model")
        return

    # Check that there's a bucket where to upload the model
    if not settings.storage_bucket:
        sl.sidebar.write(
            "⚠️ there's no bucket properly configured where to upload the classifier"
        )
        return

    # Switch to tell if it should download or upload
    upload_or_download = sl.sidebar.radio("Upload or Download", ["upload", "download"])

    def upload():
        sl.info(
            f"Uploading model in {branch_name}/{experiment_name} of type '{classifier_type}' to bucket..."
        )
        try:
            app.upload_model_of_type(experiment_name, branch_name, classifier_type)
            sl.success("Upload successful")
        except Exception as e:
            sl.error(f"Could not complete upload. Error: '{e}'")

    if upload_or_download == "upload":
        sl.sidebar.button(
            "Upload",
            help=f"Upload model in '{experiment_name}/{branch_name}' of type '{classifier_type}' to bucket {settings.storage_bucket or '<no bucket specified in environment>'}",
            on_click=upload,
        )
        return

    # Ask for a path where to download
    path = sl.sidebar.text_input("Download Path")

    def download():
        path_obj = Path(path)
        if not path or not path_obj.exists() or not path_obj.is_dir():
            sl.error(f"The path '{path}' is not a valid download path for a model")
            return

        sl.info(f"Downloading model of type' {classifier_type}' to '{path}'...")

        # try to perform download
        try:
            app.download_model_of_type(path, classifier_type)
            sl.success("Download successful")
        except Exception as e:
            sl.error(f"Could not complete download. Error: '{e}'")

    sl.sidebar.button(
        "Download",
        help=f"Download model of type '{classifier_type}' from bucket '{settings.storage_bucket}'",
        on_click=download,
    )


if backend == "local":
    upload_and_download_model()

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
    options=app.manager.available_crawlers(),
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

# Move function on cloud backend
if backend == "cloud":

    def move():
        sl.info("Moving data from firestore to big query")
        try:
            app.move()
            sl.success("Move operation succesful")
        except Exception as e:
            sl.error(f"Could not perform move operation. Error: {e}")

    sl.sidebar.write("-------")
    sl.sidebar.write("# Move")
    sl.sidebar.write("Move instances from firestore to big query")
    sl.sidebar.button("Move", on_click=move)

# -- < Show Dashboard > ---------------------------------------
sl.dataframe(
    app.get_dashboard_data(
        label_relevance=relevance,
        label_service=service,
        max_rows=max_rows,
        scraped=scraped,
    )
)
