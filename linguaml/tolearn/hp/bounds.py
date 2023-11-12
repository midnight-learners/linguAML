from typing import Self, Iterable, Any
from pydantic import BaseModel, ConfigDict, model_validator

class Bounds(BaseModel):
    
    min: float
    max: float
    
    model_config = ConfigDict(
        frozen=True
    )
    
    @classmethod
    def from_iterable(cls, bounds: Iterable) -> Self:
        """Instantiates from an iterable collection with two numbers.

        Parameters
        ----------
        bounds : Iterable
            An iterable collection with two numbers, 
            one is the lower bound, 
            and the other is the upper bound.

        Returns
        -------
        Self
            An instance of this class.
        """
        
        return cls(**cls._convert_iterable_to_fields(bounds))
    
    @model_validator(mode="before")
    @classmethod
    def preprocess_input(cls, data: Any) -> dict:
        
        if isinstance(data, dict):
            return data
        
        if not isinstance(data, Iterable):
            raise ValueError("unsupported input data")
        
        fields = cls._convert_iterable_to_fields(data)
        
        return fields
    
    @staticmethod
    def _convert_iterable_to_fields(bounds: Iterable) -> dict:
        
        # Convert to tuple
        bounds = tuple(map(float, bounds))
        assert len(bounds) == 2,\
            "the length of the bounds must be exactly 2"
        
        # Construct fields
        fields = {
            "min": min(bounds),
            "max": max(bounds)
        }
        
        return fields

class NumericHPBounds(BaseModel):
    
    name2bounds: dict[str, Bounds]
    
    model_config = ConfigDict(
        frozen=True
    )
    
    @property
    def hp_names(self) -> tuple[str]:
        """All hyperparameter names.
        """
        
        return tuple(self.name2bounds.keys())
    
    @classmethod
    def from_dict(cls, name2bounds: dict) -> Self:
        """Instantiates from a dictionary.

        Parameters
        ----------
        name2bounds : dict
            A dictionary each entry of which is like:
            - hyperparameter name: str
            - bounds: Bounds

        Returns
        -------
        Self
            An instance of this class.
        """
        
        return cls.model_validate(name2bounds)
        
    def get_bounds(self, hp_name: str) -> Bounds:
        """Get the lower and upper bounds of a hyperparameter.

        Parameters
        ----------
        hp_name : str
            Hyperparameter name.

        Returns
        -------
        Bounds
            Lower and upper bounds.
        """
        
        return self.name2bounds[hp_name]
    
    @model_validator(mode="before")
    @classmethod
    def preprocess_input(cls, data: Any) -> dict:
        
        if not isinstance(data, dict):
            raise ValueError("unsupported input data")
        
        if "name2bounds" in data:
            return data
        
        # Wrap as the value of the key "name2bounds"
        name2bounds = cls._wrap_name2bounds(data)
        
        return name2bounds
    
    @staticmethod
    def _wrap_name2bounds(name2bounds: dict) -> dict:
        
        return {
            "name2bounds": name2bounds
        }
