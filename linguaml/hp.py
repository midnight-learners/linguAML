from typing import Iterable
from abc import ABC
from pydantic import BaseModel

class HpConfig(BaseModel, ABC):
    
    @classmethod
    def dim(cls) -> int:
        """Dimension of the hyperparameter space, i.e.,
        the number of hyperparameters.
        """
        
        return len(cls.param_names())
    
    @classmethod
    def param_names(cls) -> tuple[str]:
        """All hyperparameter names.
        """
        
        return tuple(cls.__fields__.keys())
    
    @classmethod
    def numeric_param_names(cls) -> tuple[str]:
        """Names of numeric hyperparameters.
        """
        
        return tuple(filter(
            lambda param_name: cls.param_type(param_name) in (float, int),
            cls.param_names()
        ))
    
    @classmethod
    def param_type(cls, name: str) -> type:
        """Data type of the hyperparameter.
        """
        
        return cls.__fields__.get(name).type_
    
    @classmethod
    def from_action(
            cls, 
            action: Iterable[float],
            bounds: dict[str, tuple]
        ) -> tuple[str]:
        
        hps = {}
        
        # Iterate over all slots in the action
        for i, hp in enumerate(action):
            
            # Find all numberic HPs
            hp_name = cls.numeric_param_names()[i]
            
            # Get the HP type, which is either float or int
            hp_type = cls.param_type(hp_name)
            
            # Lower and upper bounds
            hp_min, hp_max = bounds[hp_name] 
            
            # Recover the HP value        
            hp = hp_type(hp * (hp_max - hp_min) + hp_min)
            
            hps[hp_name] = hp
            
        return cls(**hps)
