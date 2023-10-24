import typer
from typing_extensions import Annotated
from rich import print

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
    
    import json
    from ..config import settings
    from ..data.dataset.description import DatasetDescription

    for i, dataset_dir in enumerate(settings.data_dir.iterdir()):
        
        # Load dataset description from metadata.json file
        metadata_filepath = dataset_dir.joinpath("metadata").with_suffix(".json")
        with open(metadata_filepath, "r") as f:
            metadata = json.load(f)
        description = DatasetDescription.from_metadata(metadata)
        
        # Print line
        line = description.name
        if index:
            line = f"{i + 1} {line}"
        print(line)
