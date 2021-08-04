
from dynaconf import Dynaconf, Validator
import os

_HOME = os.environ.get("HOME")

settings = Dynaconf(
    envvar_prefix="DYNACONF",
    settings_files=['./settings.toml', 'config/.secrets.toml'],
    validators= [
        Validator("DATE_FORMAT", default="%Y-%m-%d %H:%M:%S.%f%z" ),
        Validator("C4V_FOLDER", default=os.path.join(_HOME, ".c4v")),
        Validator("LOCAL_SQLITE_DB_NAME", default="c4v.sqlite")
    ]
)

# `envvar_prefix` = export envvars with `export DYNACONF_FOO=bar`.
# `settings_files` = Load this files in the order.
