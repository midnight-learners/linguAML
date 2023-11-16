from .app import app

# Import submodules to register the commands
from .config import config_app
from .dataset import dataset_app
from .tune import *

app.add_typer(config_app, name="config")
app.add_typer(dataset_app, name="dataset")

def main():
    app()
