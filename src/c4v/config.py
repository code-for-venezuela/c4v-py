from dynaconf import Dynaconf, Validator
import os

_HOME = os.environ.get("HOME")
settings = Dynaconf(
    envvar_prefix="C4V",
    settings_files=["./settings.toml", "config/.secrets.toml"],
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
    ],
)

# `envvar_prefix` = export envvars with `export DYNACONF_FOO=bar`.
# `settings_files` = Load this files in the order.
