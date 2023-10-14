from typing import Optional
from pydantic import BaseModel, Field
from ..config import settings

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