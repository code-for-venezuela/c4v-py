"""
    The ScrapedData class is a target scheme for scraped data output, it describes how should stored in the db
"""

# Python imports
from dataclasses import dataclass, asdict, field, fields

from typing import List, Dict, Any
from datetime import datetime
import json
import enum

# Local imports
from c4v.config import settings


class Labels(enum.Enum):
    """
        Every possible label
    """

    IRRELEVANTE: str = "IRRELEVANTE"
    DENUNCIA_FALTA_DEL_SERVICIO: str = "PROBLEMA DEL SERVICIO"
    UNKNOWN: str = "UNKNOWN"

    @classmethod
    def labels(cls) -> List[str]:
        """
            Get list of labels as strings
        """
        return [l.value for l in cls]


class Sources(enum.Enum):
    """
        Every possible source
    """

    UNKOWN: str = "UNKNOWN"
    SCRAPING: str = "SCRAPING"
    CLIENT: str = "CLIENT"


@dataclass()
class ScrapedData:
    """
        This is a general data format class, 
        every data format for other scrapers could have 
        additional fields according to its needs and
        scrapable data, but then they should be able to 
        convert themselves into this format, possibly leaving a 
        few fields as None. Thus, we can be able to 
        easily map from a scrapers's output to a database 
        scheme
    """

    url: str
    last_scraped: datetime = None
    title: str = None
    content: str = None
    author: str = None
    categories: List[str] = field(default_factory=list)
    date: str = None
    label: Labels = None
    source: Sources = None

    def pretty_print(self, max_content_len: int = -1) -> str:
        """
            Return a human-readable representation of this data.
            Truncate content if requested
        """

        # create categories string
        categories = "".join(map(lambda s: f"\t+ {s}\n", self.categories))

        max_content_len = max_content_len if max_content_len > 0 else len(self.content)

        # create body content:
        content = self.content
        if max_content_len < len(self.content):
            content = content[:max_content_len] + "..."

        return f"title: {self.title}\nauthor: {self.author}\ndate: {self.date}\ncategories:\n{categories}\nlabel: {self.label}\nsource: {self.source}\ncontent:\n\t{content}"

    def __hash__(self) -> int:
        return (
            self.url,
            self.last_scraped,
            self.title,
            self.content,
            self.author,
            self.date,
            self.label,
            self.source,
        ).__hash__()

    @property
    def is_scraped(self) -> bool:
        """
            Tells if this instance was already scraped at some point
        """
        return self.last_scraped != None

    def to_dict(self) -> Dict[str, Any]:
        """
            Transform this instance to a dict
        """
        d = asdict(self)

        # Convert label to str
        label : Labels = d.get("label")
        if label:
            d['label'] = label.value
        
        # Convert source to str
        source : Sources = d.get("source")
        if source:
           d['source'] = source.value
        
        # Convert date to str
        last_scraped : datetime = d.get('last_scraped')
        if last_scraped:
            d['last_scraped'] = datetime.strftime(last_scraped, settings.date_format)

        return d

    @classmethod
    def from_dict(cls, scraped_data : Dict[str, Any] ):
        """
            Create scraped data instance from a dict
        """
        # Sanity check
        valid_fields = [x.name for x in fields(cls)]
        if any(k for k in scraped_data.keys() if k not in valid_fields):
            raise ValueError("Invalid scraped data dict representation")

        # Parse label
        label : str = scraped_data.get("label")
        if label:
            scraped_data['label'] = Labels(label)
        
        # Parse source
        source : str  = scraped_data.get("source")
        if source:
           scraped_data['source'] = Sources(source)
        
        # Parse date 
        last_scraped : str = scraped_data.get('last_scraped')
        if last_scraped:
            scraped_data['last_scraped'] = datetime.strptime(last_scraped, settings.date_format)
        
        return cls(**scraped_data)

class ScrapedDataEncoder(json.JSONEncoder):
    """
        Encoder to turn this file into json format
    """

    def default(self, obj: ScrapedData) -> Dict[str, Any]:
        if isinstance(obj, ScrapedData):
            return asdict(obj)
        elif isinstance(obj, datetime):
            return datetime.strftime(obj, settings.date_format)

        return json.JSONEncoder.default(self, obj)
