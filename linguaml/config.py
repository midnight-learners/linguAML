from typing import Self, Optional
from pathlib import Path
import tomllib
from pydantic_settings import BaseSettings
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
        
        # Read the custom configuration settings
        with open(config_filepath, "rb") as f:
            custom_settings_dict = tomllib.load(f)
            
        # Get the dict of current settings
        settings_dict = settings.model_dump()
        
        # Update the current settings dict
        settings_dict.update(custom_settings_dict)
        
        # Rebuild the settings instance
        settings = Settings.model_validate(settings_dict)

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
