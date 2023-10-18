from typing import Self, Optional
from pydantic import BaseModel

class DatasetDescription(BaseModel):
    
    name: str
    abstract: str
    summary: str
    variable_info: Optional[str] = None
    
    @classmethod
    def from_metadata(cls, metadata: dict) -> Self:
        """Instantiate from the metadata of a UCI dataset.

        Parameters
        ----------
        metadata : dict
            An attribute from the downloaded UCI dataset 
            using function `fetch_ucirepo` from package `ucimlrepo`.
            For more information, refer to https://github.com/uci-ml-repo/ucimlrepo.
            
        Returns
        -------
        Self
            An Instance of this class.
        """
        
        return cls(
            name=metadata["name"],
            abstract=metadata["abstract"],
            summary=metadata["additional_info"]["summary"],
            variable_info=metadata["additional_info"]["variable_info"]
        )
    
    def to_markdown(self) -> str:
        """Convert to Markdown content.

        Returns
        -------
        str
            Dataset description in Markdown.
        """
        
        lines = []
        
        # Title, name of the dataset
        lines.append(f"# {self.name}\n")
        
        # Abstract
        lines.append(f"{self.abstract}\n")
        
        # Summary
        lines.append("## Summary\n")
        lines.append(self.summary)
        
        # Variable information
        if self.variable_info is not None:
            lines.append("## Variable Information\n")
            lines.append(self.variable_info)
        
        md = "\n".join(lines)
        return md
