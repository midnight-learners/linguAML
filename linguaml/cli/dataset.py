import typer
from typing_extensions import Annotated
from rich import print
from ..config import settings

dataset_app = typer.Typer()

@dataset_app.command()
def list(
        index: Annotated[
            bool,
            typer.Option(
                help="Show the index of the dataset in the front of its name"
            )
        ] = False
    ) -> None:

    for i, dataset_dir in enumerate(settings.data_dir.iterdir()):
        line = dataset_dir.name
        if index:
            line = f"{i + 1} {line}"
        print(line)
