import typer
from typing import Optional
from typing_extensions import Annotated
from rich import print
import tomllib
from pathlib import Path
from .app import app
# from ..train import train
# from ..data.dataset import load_dataset
# from ..families import get_family
# from ..agent import Agent

@app.command()
def train(
        file: Annotated[
            Optional[Path],
            typer.Option(
                "--file", "-f",
                help="Configuration file of the training process"
            )
        ] = None,
        n_epcohs: Annotated[
            int,
            typer.Option(
                "--n-epochs", "-n",
                help="Number of training epochs"
            )
        ] = 10
    ) -> None:
    
    if file is not None:
        train_from_file(file)
        return
    
    print(n_epcohs)
    
def train_from_file(file: Path) -> None:
    
    # Load the training settings from file
    with open(file, "rb") as f:
        train_settings = tomllib.load(f)
        
    