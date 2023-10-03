import re
from pathlib import Path
import inflection
from .logging import get_logger

def mkdir_if_not_exists(dir: Path) -> Path:
    
    # Make dir if it does not exist
    if not dir.is_dir():
        dir.mkdir(parents=True, exist_ok=True)
    
    return dir

def dasherize(text: str) -> str:
    
    # Convert to lower case
    text = text.lower()
    
    # Replace special characters with whitespaces
    text = re.sub(r"[^a-z0-9]", " ", text).strip()
    
    # Replace all whitespaces with dashes
    text = re.sub(r"\s+", "-", text)
    
    # Finalize
    text = inflection.dasherize(text)
    
    return text

__all__ = [
    "mkdir_if_not_exists",
    "dasherize",
    "get_logger"
]