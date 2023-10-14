from typing import Optional
from pathlib import Path
from pydantic import BaseModel, Field
from pydantic_argparse import ArgumentParser
from .config import settings

class DatasetListCommand(BaseModel):

    def run(self):
        for dataset_dir in settings.data_dir.iterdir():
            print(dataset_dir.name)
            

class DatasetCommand(BaseModel):
    list: Optional[DatasetListCommand] = Field(description="List the datasets")
    
    def run(self):
        for command_name in self.__fields__:
            command = getattr(self, command_name)
            if command is not None:
                command.run()
                break

class TrainCommand(BaseModel):
    n_epochs: int = Field(default=10, description="Number of epochs to train")
    lr: float = Field(default=0.01, description="learning rate")
    

class AppArgs(BaseModel):
    
    # Subcommands
    dataset: Optional[DatasetCommand] = Field(description="Commands related to datasets")
    train: Optional[TrainCommand] = Field(description="Train the agent")
    
    def run(self):
        for command_name in self.__fields__:
            command = getattr(self, command_name)
            if command is not None:
                command.run()
                break
    
    
def run() -> None:
    
    parser = ArgumentParser(
        model=AppArgs,
        prog="linguaml",
        description="An AML Framework Powered by RL and LLMs",
        version="0.0.1",
        epilog="Under Active Development..."
    )
    
    args = parser.parse_typed_args()
    args.run()
    
    
    
    
    
    
    
    
