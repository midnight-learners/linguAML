from typing import Self, Optional
from pathlib import Path
from pydantic import BaseModel
import json
import pandas as pd
from ucimlrepo import fetch_ucirepo
from ...utils import mkdir_if_not_exists, dasherize
from ...config import settings
from .description import DatasetDescription

class DataSubset(BaseModel):
    
    features: pd.DataFrame
    targets: pd.DataFrame
    
    class Config:
        frozen = True
        arbitrary_types_allowed = True
        
class Dataset(BaseModel):
    
    description: DatasetDescription
    train: DataSubset
    valid: DataSubset
    test: DataSubset
    
    class Config:
        frozen = True
        arbitrary_types_allowed = True
    
    @property
    def name(self) -> str:
        """Dataset name.
        """
        
        return self.description.name
    
    @classmethod
    def from_uci(
            cls, 
            name: Optional[str] = None, 
            id: Optional[int] = None,
            save_data: bool = True,
            data_dir: Path | str = settings.data_dir,
        ) -> Self:
        
        # Load from UCI database
        result = fetch_ucirepo(name, id)
        
        # Metadata of the dataset
        metadata = result.metadata
        
        # Create description from metadata
        description = DatasetDescription.from_metadata(metadata)
        
        # Extract features and targets
        data = result.data
        features: pd.DataFrame = data.features
        targets: pd.DataFrame = data.targets
        
        # Create dataset instance
        dataset = cls(
            description=description,
            features=features,
            targets=targets
        )
        
        # Do not save data on disk
        if not save_data:
            return dataset
        
        # Dataset directory
        mkdir_if_not_exists(data_dir)
        dataset_dir_name = dasherize(description.name)
        dataset_dir = mkdir_if_not_exists(data_dir.joinpath(dataset_dir_name))

        # Save metadata
        dataset_metadata_filepath = dataset_dir.joinpath("metadata").with_suffix(".json")
        with open(dataset_metadata_filepath, "w") as f:
            json.dump(
                metadata,
                f,
                indent=4
            )
        
        # Write a README file
        dataset_readme_filepath = dataset_dir.joinpath("README").with_suffix(".md")
        with open(dataset_readme_filepath, "w") as f:
            f.write(description.to_markdown())
        
        # Save features and tagerts
        dataset_data_dir = mkdir_if_not_exists(dataset_dir.joinpath("data"))
        features.to_csv(
            dataset_data_dir.joinpath("features").with_suffix(".csv"),
            index=False
        )
        targets.to_csv(
            dataset_data_dir.joinpath("targets").with_suffix(".csv"),
            index=False
        )
        
        return dataset
    
    @classmethod
    def from_dataset_dir(cls, dataset_dir: Path | str) -> Self:
        """Loads dataset from the preprocessed data under the given directory.

        Parameters
        ----------
        dataset_dir : Path | str
            The directory where everything related to the dataset is stored.

        Returns
        -------
        Self
            An instance of this class.
        """
        
        # Convert to path
        dataset_dir = Path(dataset_dir)
        
        # Load metadata
        dataset_metadata_filepath = dataset_dir.joinpath("metadata").with_suffix(".json")
        with open(dataset_metadata_filepath, "r") as f:
            metadata = json.load(f)
        
        # Create description
        description = DatasetDescription.from_metadata(metadata)
        
        # Load precrossed data
        preprocessed_data_dir = dataset_dir.joinpath("preprocessed-data")
        
        # Collect each data subset
        data_subsets = {}
        for data_subset_name in ("train", "valid", "test"):
            
            # Training, valid or test data directory
            data_subset_dir = preprocessed_data_dir.joinpath(data_subset_name)

            # Get features and targets
            features_filepath = data_subset_dir.joinpath("features").with_suffix(".csv")
            targets_filepath = data_subset_dir.joinpath("targets").with_suffix(".csv")
            features = pd.read_csv(features_filepath, header=None)
            targets = pd.read_csv(targets_filepath, header=None)
            
            # Create data subset
            data_subset = DataSubset(features=features, targets=targets)
            data_subsets[data_subset_name] = data_subset
        
        return cls(
            description=description,
            train=data_subsets["train"],
            valid=data_subsets["valid"],
            test=data_subsets["test"]
        )

def load_dataset(dataset_dir: Path | str) -> Dataset:
    
    return Dataset.from_dataset_dir(dataset_dir)
