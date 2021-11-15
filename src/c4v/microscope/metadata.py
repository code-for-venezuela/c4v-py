"""
    The manager may need to store some persistent configuration data
    between runs, here we find the scheme requested for the system
"""
# Local imports
from typing import Type
from c4v.config import settings

# Python imports
import dataclasses
import json


@dataclasses.dataclass
class Metadata:
    """
        Data required and stored by the manager to perform common operations.    
        # Parameters:
            - classifier_model : `str` = Default classifier model. Could be a huggingface model or a locally stored one
            - base_language_model : `str` = Default base language model used as a base for training the new classifier model
            - persistency_manager : `str` = Persistency manager to use. One of the possible variants of the c4v.config.PersistencyManagers enum class
            - user_persistency_manager_path : `str` = Path to a file returning a custom persistency manager to use for the CLI
                                                    tool. Such file should provide a function `get_persistency_manager : () -> BasePersistencyManager` 
                                                    that will be used to retrieve the Persistency Manager object to use.    
                                                    If not provided, the default sqlite based manager is used. 
    """

    classifier_model: str = settings.default_base_language_model  # Absolute path for the model to load
    base_language_model: str = settings.default_base_language_model  # Absolute path for the base model to load
    persistency_manager: str = settings.persistency_manager  # Type of manager to use
    user_persistency_manager_path: str = settings.user_persistency_manager_path  # Path to a persistency manager to use in the CLI tool. If not provided, defaults to a SQLite one
    user_persistency_manager_module: str = settings.user_persistency_manager_module  # module where to get the persistency manager itself

    def to_json_str(self, pretty: bool = False) -> str:
        """
            Return a json-formated string representation for this object
        """
        indent = 1 if pretty else None
        return json.dumps(self, cls=_MetadataJSONEncoder, indent=indent)

    @classmethod
    def from_json_str(cls, json_str: str):
        """
            Create instance from json-formated string
        """
        return cls(**json.loads(json_str))

    def to_json(self, filename: str, pretty: bool = False) -> str:
        """
            Return a json-formated string representation for this object
        """
        indent = 1 if pretty else None
        with open(filename, "w") as f:
            return json.dump(self, f, cls=_MetadataJSONEncoder, indent=indent)

    @classmethod
    def from_json(cls, json_path: str):
        """
            Create an instance from a json file
            Parameters:
                json_path : str = path to file to load, may raise FileNotFound error if file does not exists
        """
        with open(json_path) as f:
            try:
                return cls(**json.load(f))
            except TypeError as e:  # in case of an unexpected argument
                raise TypeError(f"Couldn't parse metadata object from json. Error: {e}")


class _MetadataJSONEncoder(json.JSONEncoder):
    """
        Class to serialize metadata into a json file
    """

    def default(self, o):
        if dataclasses.is_dataclass(o):
            return dataclasses.asdict(o)
        return super().default(o)
