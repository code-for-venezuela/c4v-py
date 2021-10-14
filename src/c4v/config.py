from dynaconf import Dynaconf, Validator
import os
import enum
_HOME = os.environ.get("HOME")

class PersistencyManagers(enum.Enum):
    """
        Possible arguments for "PERSISTENCY_MANAGER" argument,
        telling the possible valid variations of persistency managers
    """
    SQLITE: str = "SQLITE"
    BIGQUERY: str = "BIG_QUERY"

settings = Dynaconf(
    envvar_prefix="C4V",
    settings_files=["settings.toml", ".secrets.toml"],
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
            "PERSISTENCY_MANAGER", default=PersistencyManagers.SQLITE.value, is_in=[x.value for x in PersistencyManagers]
        )
    ],
    load_dotenv=True
)

# `envvar_prefix` = export envvars with `export DYNACONF_FOO=bar`.
# `settings_files` = Load this files in the order.
