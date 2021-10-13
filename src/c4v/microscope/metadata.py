"""
    The manager may need to store some persistent configuration data
    between runs, here we find the scheme requested for the system
"""
# Local imports
from c4v.config import settings

# Python imports
import dataclasses
import json


@dataclasses.dataclass
class Metadata:
    """
        Data required and stored by the manager to perform common operations
    """

    classifier_model: str = settings.default_base_language_model  # Abosolute path for the model to load
    base_language_model: str = settings.default_base_language_model  # Abosolute path for the base model to load

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
            return cls(**json.load(f))


class _MetadataJSONEncoder(json.JSONEncoder):
    """
        Class to serialize metadata into a json file
    """

    def default(self, o):
        if dataclasses.is_dataclass(o):
            return dataclasses.asdict(o)
        return super().default(o)
