"""
    This file contains a class for the app configuration.
    It gets configured once the app starts using the environment 
    variables as a configuration
"""

# Python imports
import os
import enum
from typing import List

class EnvOptions(enum.Enum):
    """
        Available environment configuration variables
    """
    C4V_DB_BACKEND : str = "C4V_DB_BACKEND" 

class DBBackEndOptions(enum.Enum):
    """
        Available DB back end options
    """
    SQLITE : str = "SQLITE"

    @classmethod
    def valid_values(cls) -> List[str]:
        return [x.value for x in cls]

class Config:
    """
        App configuration State
    """

    # db to use when requesting for stored data
    default_db_backend : str = DBBackEndOptions.SQLITE.value

    def __init__(self, db_backend : str = None):

        # Set up db backend. 
        # If no backend is provided, try to find one in the environment. 
        # If not provided there, then use the default one 
        if db_backend:
            self._db_backend =  db_backend
        else:
            # Get configuration from environmen
            env_backend = os.environ.get(EnvOptions.C4V_DB_BACKEND.value)
            if env_backend and env_backend not in DBBackEndOptions.valid_values():
                raise ValueError(   f"Invalid environment configuration for Back End DB. Given value: {env_backend}\n" + \
                                    f"Possible values: {DBBackEndOptions.valid_values()}" 
                            )
            self._db_backend = env_backend
        
    @property
    def db_backend(self) -> str:
        """
            Configured db backend
        """
        return self._db_backend or self.default_db_backend
