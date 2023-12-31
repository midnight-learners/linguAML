import re
from pathlib import Path
import inflection

def mkdir_if_not_exists(dir: Path) -> Path:
    
    # Make dir if it does not exist
    if not dir.is_dir():
        dir.mkdir(parents=True, exist_ok=True)
    
    return dir

def dasherize(text: str) -> str:
    
    # Convert camel case to snake case
    text = inflection.underscore(text)
    
    # Convert to lower case
    text = text.lower()
    
    # Replace special characters with whitespaces
    text = re.sub(r"[^a-z0-9]", " ", text).strip()
    
    # Replace all whitespaces with dashes
    text = re.sub(r"\s+", "-", text)
    
    return text
