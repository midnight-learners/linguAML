from .app import app
from .config import config_app
from .dataset import dataset_app
from .train import *

app.add_typer(config_app, name="config")
app.add_typer(dataset_app, name="dataset")

def main():
    app()
