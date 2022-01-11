"""
    In this module you will find multiple configurations for the app. You can override them by specifying a different
    .env file, exporting environment variables properly named, or providing a settings.toml file.

    Remember that .env files should be in the project's root.
"""

from dynaconf import Dynaconf, Validator
import os
import enum

_HOME = os.environ.get("HOME")


class PersistencyManagers(enum.Enum):
    """
        Possible arguments for "PERSISTENCY_MANAGER" setting,
        telling the possible valid variations of persistency managers.
        # Variations:
            - `SQLITE` = the default SQLite based persistency manager
            - `GCLOUD`  = use a google cloud based persistency manager, measurements are stored in cloud
            - `USER` = Use the user defined persistency manager, specified by the USER_PERSISTENCY_MANAGER_PATH setting.
    """

    SQLITE: str = "SQLITE"
    GCLOUD: str  = "GCLOUD"  
    USER: str = "USER"

settings = Dynaconf(
    envvar_prefix="C4V",
    settings_files=["./settings.toml", "config/.secrets.toml"],
    load_dotenv=True,
    validators=[
        Validator("DATE_FORMAT", default="%Y-%m-%d %H:%M:%S.%f%z"),
        Validator("C4V_FOLDER", default=os.path.join(_HOME, ".c4v")),
        Validator(
            "LOCAL_SQLITE_DB_NAME", default="c4v_db.sqlite"
        ),  # Path to the local sqlite db file
        Validator("LOCAL_SQLITE_DB", default=os.path.join(_HOME, ".c4v/c4v_db.sqlite")),
        Validator(
            "DEFAULT_BASE_LANGUAGE_MODEL", default="BSC-TeMU/roberta-base-bne"
        ),  # Base language model for the classifier
        Validator(
            "DEFAULT_LANG_MODEL_MIN_LOSS", default=0.15
        ),  # Minimum acceptable loss for base language model, if the loss is greater, a new training is required
        Validator(
            "EXPERIMENTS_DIR", default=os.path.join(_HOME, ".c4v/experiments/")
        ),  # Default name for experiments folder
        Validator(
            "PERSISTENCY_MANAGER",
            default=PersistencyManagers.SQLITE.value,
            is_in=[x.value for x in PersistencyManagers],
        ),  # Specify which persistency manager to use, possible options defined by the PersistencyManagers enum class.
        Validator(
            "USER_PERSISTENCY_MANAGER_PATH", default=None
        ),  # Path for the cusmtom user persistency manager python file. Should export a function: `get_persistency_manager : () -> BasePersistencyManager`
        Validator(
            "USER_PERSISTENCY_MANAGER_MODULE", default=None
        ),  # Module from the imported file where to find the the function
        Validator(
            "CLI_LOGGING_LEVEL", default=10
            # Logging level:
            #   - 10: error, warn, success, info
            #   - 9:  error, warn, success
            #   - 8:  error, warn
            #   - 7:  error
            #   - [1-6]:  reserved for future use
            #   - 0: No logging 
        ),  
        Validator(
            "STORAGE_BUCKET", default=None,
            # Used to choose the name of the bucket where data like the classifier models is stored
        ),
        Validator(
            "SCRAPED_DATA_TABLE", default=None,
            # Used to select the table where to retrieve data from in google cloud
        ),
        Validator(
            "SCRAPING_CLOUD_URL_TRIGGER", default=None
            # Url to use to trigger a scaping process in a google cloud function
        ),
        Validator(
            "CRAWLING_CLOUD_URL_TRIGGER", default=None
            # Url to use to trigger a crawling process in a google cloud function
        ),
        Validator(
            "CLASSIFY_CLOUD_URL_TRIGGER", default=None
            # Url to use to trigger a classification process in a google cloud function
        ),
        Validator(
            "GCLOUD_PROJECT_ID", default=None
            # project id for the gcloud project, where all the cloud operations will be performed
        ),
        Validator(
            "GCLOUD_MAX_CONTENT_LEN", default=6000
        )
    ]
)

