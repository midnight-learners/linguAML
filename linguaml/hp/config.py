from typing import Iterable
from abc import ABC
from enum import Enum
from pydantic import BaseModel
from .cat import CategoricalHP

class HPConfig(BaseModel, ABC):
    
    @classmethod
    def n_hps(cls) -> int:
        """Number of hyperparameters.
        """
        
        return len(cls.hp_names())
    
    @classmethod
    def n_numeric_hps(cls) -> int:
        """Number of numeric hyperparameters.
        """
        
        return len(cls.numeric_hp_names())
    
    @classmethod
    def n_categorical_hps(cls) -> int:
        """Number of categorical hyperparameters.
        """
        
        return len(cls.categorical_hp_names())
    
    @classmethod
    def hp_names(cls) -> tuple[str]:
        """All hyperparameter names.
        """
        
        return tuple(cls.model_fields.keys())
    
    @classmethod
    def numeric_hp_names(cls) -> tuple[str]:
        """Names of numeric hyperparameters.
        """
        
        return tuple(filter(
            lambda param_name: cls.hp_type(param_name) in (float, int),
            cls.hp_names()
        ))
    
    @classmethod
    def categorical_hp_names(cls) -> tuple[str]:
        """Names of categorical hyperparameters.
        """
        
        return tuple(filter(
            lambda param_name: issubclass(cls.hp_type(param_name), Enum),
            cls.hp_names()
        ))
    
    @classmethod
    def hp_type(cls, name: str) -> float | int | type[CategoricalHP]:
        """Data type of the hyperparameter.
        """
        
        return cls.model_fields.get(name).annotation
    
    @classmethod
    def n_levels_in_category(cls, categorical_hp_name: str) -> int:
        """Number of levels in the given category.

        Parameters
        ----------
        category : str
            Categorical hyperparameter name.

        Returns
        -------
        int
            Number of levels.
        """
        
        category_type: CategoricalHP = cls.hp_type(categorical_hp_name)
        
        return category_type.n_levels()
    
    @classmethod
    def description(cls) -> dict[str, str]:
        """Returns a dictionary that maps
        each hyperparameter name to its description.

        Returns
        -------
        dict[str, str]
            Each item is like:
            <hyperparameter name>: <description>
        """
        
        hp_name_to_description = {
            hp_name: field_info.description
            for hp_name, field_info in cls.model_fields.items()
        }
        
        return hp_name_to_description
    