import typer
from typing_extensions import Annotated
from rich import print
from .. import __version__

APP_NAME = "LinguAML"

app = typer.Typer(
    help="An AML Framework Powered by RL and LLMs",
)

def version_callback(value: bool) -> None:
    
    if value:
        print(f"{APP_NAME} Version: {__version__}")
        raise typer.Exit()

@app.callback()
def common(
    ctx: typer.Context,
    version: Annotated[
        bool,
        typer.Option(
            "--version", "-V",
            callback=version_callback,
            help="Show the version and exits.",
        )
    ] = False
) -> None:
    pass
