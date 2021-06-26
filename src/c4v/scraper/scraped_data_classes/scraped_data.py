from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
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

    title: str = None
    content: str = None
    author: str = None
    categories: List[str] = None
    date: str = None

    def pretty_print(self, max_content_len : int = -1) -> str:
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