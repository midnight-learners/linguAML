from typing import Self, Optional
from pathlib import Path
import tomllib
from pydantic import BaseSettings
from xpyutils import singleton

PROJECT_STORAGE_DIR = Path.home().joinpath(".linguaml")

# Existing configuration file
# under the project storage directory
EXISTING_CONFIG_FILENAME = ".conf"
EXISTING_CONFIG_FILEPATH = PROJECT_STORAGE_DIR.joinpath(EXISTING_CONFIG_FILENAME)

@singleton
class Settings(BaseSettings):
    
    data_dir: Path = PROJECT_STORAGE_DIR.joinpath("data")
    log_filepath: Path = PROJECT_STORAGE_DIR.joinpath(".log")
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
    
    @classmethod
    def from_exisiting_file(cls) -> Self:
        
        # The existing configuration file
        # uder the project storage directory
        existing_config_filepath = PROJECT_STORAGE_DIR.joinpath(EXISTING_CONFIG_FILENAME)
        
        # Load from existing file
        if existing_config_filepath.is_file():
            return cls.from_file(existing_config_filepath)
        
        # Otherwise, just return the default settings
        return cls()
        

def load_config(config_filepath: Path | str) -> Settings:
    
    return Settings.from_file(config_filepath)

settings = Settings.from_exisiting_file()
