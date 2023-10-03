from typing import Self, Optional
from pathlib import Path
import tomllib
from pydantic import BaseSettings
from xpyutils import singleton

@singleton
class Settings(BaseSettings):
    
    project_dir: Path = Path.home().joinpath(".linguaml")
    data_dir: Path = project_dir.joinpath("data")
    log_filepath: Path = project_dir.joinpath(".log")
    openai_api_key: Optional[str] = None
    
    @classmethod
    def from_file(cls, config_filepath: Path | str) -> Self:
        
        # Create the one and only settings
        settings = cls()
        
        with open(config_filepath, "rb") as f:
            custom_settings = tomllib.load(f)
            
        for key in custom_settings:
            if key in settings.__fields__:
                field_type = cls.__fields__.get(key).type_
                value = field_type(custom_settings[key])
                setattr(settings, key, value)

        return settings

def load_config(config_filepath: Path | str) -> Settings:
    
    return Settings.from_file(config_filepath)

settings = Settings()
