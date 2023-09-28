from typing import Self
from pathlib import Path
import tomllib
from pydantic import BaseSettings
from xpyutils import singleton
from .utils import mkdir_if_not_exists

@singleton
class Settings(BaseSettings):
    
    project_dir: Path = mkdir_if_not_exists(Path.home().joinpath(".linguaml"))
    data_dir: Path = mkdir_if_not_exists(project_dir.joinpath("data"))
    
    @classmethod
    def from_toml(cls, config_filepath: Path | str) -> Self:
        
        # Create the one and only settings
        settings = cls()
        
        with open(config_filepath, "rb") as f:
            custom_settings = tomllib.load(f)
            
        if "data_dir" in custom_settings:
            settings.data_dir = Path(custom_settings["data_dir"])
        
        return settings

def load_config(config_filepath: Path | str) -> Settings:
    
    return Settings.from_toml(config_filepath)

settings = Settings()