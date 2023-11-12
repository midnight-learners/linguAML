from typing import Self
from numpy import ndarray
import numpy as np
from torch import Tensor

# Imports from this package
from linguaml.types import Number, Vector, check_vector
from linguaml.tolearn.families.base import Family
from linguaml.tolearn.hp import HPConfig, CategoricalHP
from linguaml.tolearn.hp.bounds import NumericHPBounds
from .action import Action

class BatchedActions(dict):
    
    def __init__(self, *args, **kwargs) -> None:
        
        # Initialize super class
        super().__init__(*args, **kwargs)
    
    def __setitem__(
            self, 
            hp_name: str, 
            values: Vector | ndarray | Tensor
        ) -> None:
        
        # Convert value to a number
        if check_vector(values):
            values = np.array(values)
        elif isinstance(values, ndarray):
            assert len(values.shape) == 1,\
                "The value must be a one-dimensional array"
            values = values
        elif isinstance(values, Tensor):
            assert len(values.shape) == 1,\
                "The values must be a one-dimensional tensor"
            values = values.detach().numpy()
        else:
            raise TypeError(f"Unsupported type: {type(values)}")
            
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
    
    def to_actions(self) -> list[Action]:
        
        actions = []
        
        n_batches = len(next(iter(self.values())))
        for i in range(n_batches):
            hp_name_to_value =  {hp_name: values[i] for hp_name, values in self.items()}
            action = Action.from_dict(hp_name_to_value)
            actions.append(action)
            
        return actions
    
    def to_hp_configs(
            self, 
            family: Family, 
            numeric_hp_bounds: NumericHPBounds
        ) -> HPConfig:
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
        
        return list(map(
            lambda action: action.to_hp_config(family, numeric_hp_bounds),
            self.to_actions()
        ))
