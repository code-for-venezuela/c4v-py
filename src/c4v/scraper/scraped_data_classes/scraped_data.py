# Python imports
from dataclasses import dataclass, asdict, field
from typing      import List, Dict, Any
from datetime    import datetime
from c4v.config  import settings
import json


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

        return f"title: {self.title}\nauthor: {self.author}\ndate: {self.date}\ncategories:\n{categories}content:\n\t{content}"

    def __hash__(self) -> int:
        return (self.url, self.last_scraped, self.title, self.content, self.author, self.date).__hash__()

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
