"""
    The ScrapedData class is a target scheme for scraped data output, it describes how should stored in the db
"""

# Python imports
from dataclasses import dataclass, asdict, field, fields

from typing import List, Dict, Any, Type
from datetime import datetime
import json
import enum

# Local imports
from c4v.config import settings

class LabelSet(enum.Enum):
    """
        Interface for sets of labels that can be attached to to a model for classification
    """

    @classmethod
    def num_labels(cls) -> int:
        """
            Ammount of labels to use during classification. 
            This may be needed in case that the measurement set is using some 
            special labels that are not used for classification
        """
        return len(list(cls.get_id2label_dict().items()))

    @classmethod
    def get_id2label_dict(cls) -> Dict[int, str]:
        """
            Get a dict mapping from ids to a str, representing the labels for this label set
        """
        raise NotImplementedError("Should implement abstract method get_id2label_dict")

    @classmethod
    def get_label2id_dict(cls) -> Dict[str, int]:
        """
            Get a dict mapping labels to int ids, representing the the ids of each label in this labelset
        """
        id2label = cls.get_id2label_dict()
        return { v:k for (k,v) in id2label.items() }

class RelevanceClassificationLabels(LabelSet):
    """
        Labels for Binary classification, telling if a data instance is relevant or not
    """
    IRRELEVANTE: str = "IRRELEVANTE"
    DENUNCIA_FALTA_DEL_SERVICIO: str = "PROBLEMA DEL SERVICIO"
    UNKNOWN: str = "UNKNOWN"

    @classmethod
    def get_id2label_dict(cls) -> Dict[int, str]:
        return {
            0: cls.IRRELEVANTE.value,
            1: cls.DENUNCIA_FALTA_DEL_SERVICIO.value,
        }

class ServiceClassificationLabels(LabelSet):
    """
        Labels for service classification, "electricity", "water", "internet"... 
    """
    AGUA : str = "AGUA"
    ASEO_URBANO : str = "ASEO URBANO"
    COMBINACION : str = "COMBINACIÓN"
    ELECTRICIDAD : str = "ELECTRICIDAD"
    GAS_DOMESTICO : str = "GAS DOMÉSTICO"
    TELECOMUNICACIONES : str = "TELECOMUNICACIONES"
    NO_SERVICIO : str = "NO ES SERVICIO"
    UNKNOWN : str = "UNKNOWN"

    @classmethod
    def get_id2label_dict(cls) -> Dict[int, str]:
        return {
            0 : cls.AGUA.value,
            1 : cls.ASEO_URBANO.value,
            2 : cls.COMBINACION.value,
            3 : cls.ELECTRICIDAD.value,
            4 : cls.GAS_DOMESTICO.value,
            5 : cls.TELECOMUNICACIONES.value,
            6 : cls.NO_SERVICIO.value,
            7 : cls.UNKNOWN.value,
        }
    

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
    label_relevance: RelevanceClassificationLabels = None
    label_service: ServiceClassificationLabels = None
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

        return f"title: {self.title}\nauthor: {self.author}\ndate: {self.date}\ncategories:\n{categories}\nlabel: {self.label_relevance}\nsource: {self.source}\ncontent:\n\t{content}"

    def __hash__(self) -> int:
        return (
            self.url,
            self.last_scraped,
            self.title,
            self.content,
            self.author,
            self.date,
            self.label_relevance,
            self.label_service,
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

        # Convert label relevance to str
        label_relevance : RelevanceClassificationLabels = d.get("label_relevance")
        if label_relevance:
            d['label_relevance'] = label_relevance.value

        # Convert label service to str
        label_service : ServiceClassificationLabels = d.get("label_service")
        if label_service:
            d['label_service'] = label_service.value
        
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
            raise ValueError(f"Invalid scraped data dict representation. Invalid fields: {[k for k in scraped_data.keys() if k not in valid_fields]}")

        # Parse relevance label
        label_relevance : str = scraped_data.get("label_relevance")
        if label_relevance:
            scraped_data['label_relevance'] = RelevanceClassificationLabels(label_relevance)
        
        # Parse service label
        label_service : str = scraped_data.get("label_service")
        if label_service:
            scraped_data['label_service'] = ServiceClassificationLabels(label_service)

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
