from typing import Self
from numpy import ndarray
import numpy as np
import torch
from torch import Tensor

# Imports from this package
from linguaml.types import Number, NumberList, is_number_list
from linguaml.tolearn.family import Family
from linguaml.tolearn.hp import HPConfig, CategoricalHP
from linguaml.tolearn.hp.bounds import NumericHPBounds
from .action import ActionConfig, Action

class BatchedActions(ActionConfig, dict):
    
    def __init__(self, *args, **kwargs) -> None:
        
        # Initialize super class
        super().__init__(*args, **kwargs)
    
    def __setitem__(
            self, 
            hp_name: str, 
            values: NumberList | ndarray | Tensor
        ) -> None:
        
        # Convert value to a number
        if is_number_list(values):
            values = np.array(values)
        elif isinstance(values, ndarray):
            # Flatten the array
            values = values.flatten()
        elif isinstance(values, Tensor):
            # Flatten the tensor
            values = values.detach().flatten().numpy()
        else:
            raise TypeError(f"Unsupported type of action values: {type(values)}")
            
        super().__setitem__(hp_name, values)
        
    @classmethod
    def from_dict(cls, hp_name_2_values: dict[str, ndarray]) -> Self:
        """Construct batched actions from a dictionary.

        Parameters
        ----------
        hp_name_2_value : dict[str, ndarray]
            A dictionary mapping hyperparameter names to values (one-dimensional arrays).

        Returns
        -------
        Self
            Batched actions.
        """
        
        return cls(hp_name_2_values)
    
    @classmethod
    def from_actions(cls, actions: list[Action]) -> Self:
        """Construct batched actions from a list of actions.

        Parameters
        ----------
        actions : list[Action]
            A list of actions.

        Returns
        -------
        Self
            Batched actions.
        """
        
        # Create an empty batched actions
        batched_actions = cls()
        
        # Iterate over hyperparameter names
        for hp_name in cls.family.hp_names():
            values = [action[hp_name] for action in actions]
            batched_actions[hp_name] = values
        
        return batched_actions
    
    def to_actions(self) -> list[Action]:
        
        # Actions
        actions = []
        
        # Number of batches
        n_batches = len(next(iter(self.values())))
        
        # Iterate over batches
        for i in range(n_batches):
            hp_name_to_value =  {hp_name: values[i] for hp_name, values in self.items()}
            action = Action.from_dict(hp_name_to_value)
            actions.append(action)
            
        return actions
    
    def to_hp_configs(self) -> HPConfig:
        """Convert the action to an HPConfig instance.

        Returns
        -------
        HPConfig
            Hyperparameter configuration.
        """
        
        return list(map(
            lambda action: action.to_hp_config(),
            self.to_actions()
        ))
    
    def to_tensor_dict(self) -> dict[str, Tensor]:
        """Convert the batched actions to a dictionary of tensors.

        Returns
        -------
        dict[str, Tensor]
            A dictionary mapping hyperparameter names to tensors.
        """
        
        return {
            hp_name: torch.tensor(values, dtype=torch.float32)
            for hp_name, values in self.items()
        }
