from typing import Self, TypeAlias
from numpy import ndarray
from torch import Tensor

from linguaml.families.base import Family
from linguaml.hp.bounds import NumericHPBounds
from linguaml.hp import HPConfig, CategoricalHP

Number: TypeAlias = float | int

class Action(dict):
    
    def __init__(self, *args, **kwargs):
        
        # Initialize super class
        super().__init__(*args, **kwargs)
    
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
        
        return cls(hp_name_2_value)
        
    def to_hp_config(
            self, 
            family: Family, 
            numeric_hp_bounds: NumericHPBounds
        ) -> HPConfig:
        
        # Create a dictionary to store hyperparameters
        hps = {}
        
        # Restore numeric hyperparameters
        for hp_name in family.hp().numeric_hp_names():
            bounds = numeric_hp_bounds.get_bounds(hp_name)
            hps[hp_name] = (bounds.max - bounds.min) * self[hp_name] + bounds.min
            
        # Restore categorical hyperparameters
        for hp_name in family.hp().categorical_hp_names():
            level_index = self[hp_name]
            categorical_hp_type: type[CategoricalHP] = family.hp().hp_type(hp_name)
            categorical_hp = categorical_hp_type.from_index(level_index)
            hps[hp_name] = categorical_hp
            
        # Create an HPConfig instance
        hp_config = family.hp()(**hps)
        
        return hp_config