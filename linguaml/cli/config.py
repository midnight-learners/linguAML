import typer
from typing_extensions import Annotated
from rich import print
import tomllib
from pathlib import Path
from ..config import EXISTING_CONFIG_FILEPATH

config_app = typer.Typer(
    help="Manage the project configuration settings."
)

@config_app.command()
def set(
        file: Annotated[
            Path, 
            typer.Argument(
                show_default=False,
                help="Configuration file path"
            )
        ],
        show: Annotated[
            bool,
            typer.Option(
                help="Show the configuration settings."
            )
        ] = False
    ) -> None:
    """Update the new configuration settings by copying
    the provided configuration file
    to the project storage directory.
    """
    
    # Parse the file to examine whether it is valid
    with open(file, "rb") as f:
        settings = tomllib.load(f)
    
    # Read raw file content
    file_content = file.read_bytes()
    
    # Make the parent dir of the destination file path
    # if does not exist
    Path.mkdir(EXISTING_CONFIG_FILEPATH.parent, parents=True, exist_ok=True)
    
    # Copy the file to destination
    EXISTING_CONFIG_FILEPATH.write_bytes(file_content)
    
    # Show the configuration settings if required
    if show:
        print("Provided configuration settings:")
        print(settings)

@config_app.command()
def show():
    """Show the current configuration settings.
    """
    
    # Warn the user the there are no configuration settings
    if not EXISTING_CONFIG_FILEPATH.is_file():
        print("[bold red]Currently, there are no configuration settings![/bold red]")
        print(
            (
                "Please use the {command} command to set the configuration, "
                "or manually create a file {filepath}."
            ).format(
                command="[green]config set[/green]",
                filepath=EXISTING_CONFIG_FILEPATH
            )
        )
        return
    
    # Read the existing configuration file
    with open(EXISTING_CONFIG_FILEPATH, "rb") as f:
        settings = tomllib.load(f)

    # Show the existing configuration settings
    print(settings)
