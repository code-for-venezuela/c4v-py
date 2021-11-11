"""
    Module with some helper functions for the manager class
"""
# Local imports
import pathlib
from typing import Type
from c4v.scraper.persistency_manager.base_persistency_manager import (
    BasePersistencyManager,
)

# Python imports
from pathlib import Path
import importlib.util


def _load_user_manager(module_name: str, path: str) -> BasePersistencyManager:
    """
        Loads the user defined manager given the filepath of its python code.
        # Parameters
            - module_name : str = name of the module beeing imported
            - path : `str` = path to python file containing module to import
        # Raises
            - `FileNotFoundError` on file not existing
            - `AttributeError` on `get_persistency_manager` function not found
            - `TypeError` when the returned persistency manager is not a valid implementation of the `BasePersistencyManager` class

    """

    # Error checking
    if not Path(path).exists():
        raise FileNotFoundError(f"{path} is not a valid path, it does not exists")

    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    # Try to retrieve persistency manager
    try:
        persistency_manager = module.get_persistency_manager()
    except AttributeError as e:
        raise AttributeError(
            f"module {module} in path {path} does not provides a 'get_persistency_manager : () -> BasePersistencyManager' function. "
            + f"You should do so in order to set up your custom persistency manager object. Note that such object should implement the BasePersistencyManager base class. "
            + f"\n\tError: {e}"
        )

    # Some static checking for the new persistency manager object
    if not isinstance(persistency_manager, BasePersistencyManager):
        raise TypeError(
            f"Object returned by the `get_persistency_manager` doesn't seems to be a valid persistency manager implementation"
        )

    # Return obtained manager
    return persistency_manager
