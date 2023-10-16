from typing import Optional
from pathlib import Path
import json
from ...config import settings
from .description import DatasetDescription
from .dataset import Dataset

def get_dataset_from_name(name: str) -> Optional[Dataset]:
    
    # Get the dataset directory
    for dataset_dir in settings.data_dir.iterdir():
        
        # Get the metadata
        metadata_filepath = dataset_dir.joinpath("metadata").with_suffix(".json")
        with open(metadata_filepath, "r") as f:
            metadata = json.load(f)
            
        # Create dataset description from the metadata
        description = DatasetDescription.from_metadata(metadata)
        
        # Find the desired dataset by name
        if description.name == name:
            dataset = Dataset.from_dataset_dir(dataset_dir)
            return dataset
    
    # No dataset is found
    return None

def load_dataset(
        *, 
        name: Optional[str] = None,  
        dataset_dir: Optional[Path | str] = None
    ) -> Dataset:
    
    # Get the dataset by name
    if name is not None:
        dataset = get_dataset_from_name(name)
        return dataset
    
    # Load the dataset from its directory
    dataset = Dataset.from_dataset_dir(dataset_dir)
    
    return dataset
    