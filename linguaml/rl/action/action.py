from typing import Self, Optional
from numpy import ndarray
import numpy as np
import torch
from torch import Tensor

# Imports from this package
from linguaml.types import Number
from .config import ActionConfig
from linguaml.tolearn.hp import HPConfig, CategoricalHP

class Action(ActionConfig, dict):
    
    def __init__(self, *args, **kwargs):
        
        # Initialize super class
        super().__init__(*args, **kwargs)
        
        # Check if the class attributes are set
        self.check_attributes()        
    
    def __setitem__(
            self, 
            hp_name: str, 
            value: Number | ndarray | Tensor
        ) -> None:
        
        # Convert value to a number
        if isinstance(value, Number):
            value = value
        elif isinstance(value, ndarray):
            value = value.item()
        elif isinstance(value, Tensor):
            value = value.detach().item()
        else:
            raise TypeError(f"Unsupported type: {type(value)}")
            
        super().__setitem__(hp_name, value)
        
    @classmethod
    def from_dict(cls, hp_name_2_value: dict[str, Number]) -> Self:
        """Construct an action from a dictionary.

        Parameters
        ----------
        hp_name_2_value : dict[str, Number]
            A dictionary mapping hyperparameter names to values.

        Returns
        -------
        Self
            An action.
        """
        
        return cls(hp_name_2_value)
    
    @classmethod
    def from_hp_config(cls, hp_config: HPConfig) -> Self:
        """Construct an action from a hyperparameter configuration.

        Parameters
        ----------
        hp_config : HPConfig
            Hyperparameter configuration.

        Returns
        -------
        Self
            An action.
        """
        
        # Create an empty action
        action = cls()
        
        # Continuous part
        # Normalization of numerical hyperparameters
        # Each numerical hyperparameter is normalized to [0, 1]
        for hp_name in hp_config.numeric_hp_names():
            hp = getattr(hp_config, hp_name)
            bounds = cls.numeric_hp_bounds.get_bounds(hp_name)
            value = (hp - bounds.min) / (bounds.max - bounds.min)
            action[hp_name] = value
        
        # Discrete part
        # Each categorical hyperparameter is represented by its level index
        for hp_name in hp_config.categorical_hp_names():
            hp_value: CategoricalHP = getattr(hp_config, hp_name)
            action[hp_name] = hp_value.level_index
            
        return action
    
    @classmethod
    def random(cls, random_state: Optional[int] = None) -> Self:
        
        # Create a random generator
        rng = np.random.RandomState(seed=random_state)
        
        # Create an empty action
        action = cls()
        
        # Continuous actions
        for hp_name in cls.family.numeric_hp_names():
                
            # Generate a random number in [0, 1]
            action[hp_name] = rng.rand()
            
        # Discrete actions
        for hp_name in cls.family.categorical_hp_names():
            
            # Get the number of levels in the category
            n_levels = cls.family.n_levels_in_category(hp_name)
            
            # Generate a random integer in [0, n_levels - 1]
            action[hp_name] = np.random.randint(n_levels)
        
        return action
        
    def to_hp_config(self) -> HPConfig:
        """Convert the action to an HPConfig instance.

        Parameters
        ----------
        family : Family
            Model family.
            
        numeric_hp_bounds : NumericHPBounds
            Bounds of numeric hyperparameters.

        Returns
        -------
        HPConfig
            Hyperparameter configuration.
        """
        
        # Create a dictionary to store hyperparameters
        hps = {}
        
        # Restore numeric hyperparameters
        for hp_name in self.family.hp().numeric_hp_names():
            bounds = self.numeric_hp_bounds.get_bounds(hp_name)
            hps[hp_name] = (bounds.max - bounds.min) * self[hp_name] + bounds.min
            
        # Restore categorical hyperparameters
        for hp_name in self.family.hp().categorical_hp_names():
            level_index = self[hp_name]
            categorical_hp_type: type[CategoricalHP] = self.family.hp().hp_type(hp_name)
            categorical_hp = categorical_hp_type.from_index(level_index)
            hps[hp_name] = categorical_hp
            
        # Create an HPConfig instance
        hp_config = self.family.hp()(**hps)
        
        return hp_config

    def to_tensor_dict(self) -> dict[str, Tensor]:
        """Convert the action to a dictionary of tensors.

        Returns
        -------
        dict[str, Tensor]
            A dictionary mapping hyperparameter names to tensors.
        """
        
        return {
            hp_name: torch.tensor(value, dtype=torch.float32)
            for hp_name, value in self.items()
        }
