from pathlib import Path

def mkdir_if_not_exists(dir: Path) -> Path:
    
    # Make dir if it does not exist
    if not dir.is_dir():
        dir.mkdir(parents=True, exist_ok=True)
    
    return dir
