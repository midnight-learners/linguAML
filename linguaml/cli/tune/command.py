import typer
from typing import Optional, Annotated
from rich import print
import tomllib
from pathlib import Path

# Imports from this package
from ..app import app
from linguaml.types import TunerRole

@app.command()
def tune(
        files: Annotated[
            list[Path],
            typer.Argument(
                help="Configuration files of the tuning process."
            )
        ],
        log_filepath: Annotated[
            Optional[Path],
            typer.Option(
                "--log", "-l",
                help="File path of the tuning log to generate."
            )
        ] = None,
        role: Annotated[
            TunerRole,
            typer.Option(
                "--role", "-r",
                help="Role of the tuner."
            )
        ] = TunerRole.RL
    ) -> None:
    
    # Log file will be put under the same parent directory
    # as the tuning configuration file
    # if it is not specified
    if log_filepath is None:
        # Check all tuning configuration files share the same parent
        parents = set([file.parent for file in files])
        assert len(parents) == 1,\
            "All configuration files must be placed under the same directory"
        parent = parents.pop()
        
        # Set the log file path
        log_filepath = parent.joinpath(".log")
        
   
    
    # Load the training settings from files
    tuning_settings = {}
    for file in files:
        with open(file, "rb") as f:
            more_tuning_settings = tomllib.load(f)
        tuning_settings.update(more_tuning_settings)
    
    # Tune the machine learning models with specified tuner
    match role:
        
        case TunerRole.RL:
            # Set the log file path
            from linguaml.logger import set_log_filepath
            from linguaml.tuners.rl_tuner import logger
            set_log_filepath(logger, log_filepath)

            # Tune
            from .rl import TuningSettings, tune
            tuning_settings = TuningSettings.model_validate(tuning_settings)
            tune(tuning_settings)
        
        case TunerRole.LLM:
            # Set the log file path
            from linguaml.logger import set_log_filepath
            from linguaml.tuners.llm_tuner import logger
            set_log_filepath(logger, log_filepath)
            
            # Tune
            from .llm import TuningSettings, tune
            tuning_settings = TuningSettings.model_validate(tuning_settings)
            tune(tuning_settings)
        
        case TunerRole.HYBRID:
            pass

