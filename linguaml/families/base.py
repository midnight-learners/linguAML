from typing import Self, Optional, Iterable
from abc import ABC, abstractmethod
import numpy as np
from ..hp import HpConfig

class Predictor(ABC):
    
    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> Self:
        pass
    
    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        pass

class BaseFamily(ABC):
    
    model_type: type
    hp_config_type: type
    
    @classmethod
    def hp(cls) -> type:
        
        return cls.hp_config_type
    
    @classmethod
    def n_hps(cls) -> int:
        
        return cls.hp_config_type.dim()
    
    @classmethod
    def define_model(
            cls, 
            hp_config: Optional[HpConfig] = None,
            *,
            action: Optional[Iterable[float]] = None,
            hp_bounds: Optional[dict[str, tuple]] = None,
            **kwargs
        ) -> Predictor:
        
        if hp_config is not None:
            model = cls._define_model_from_hp_config(hp_config)
            return model
        
        if action is None and hp_bounds is None:
            model = cls.model_type(**kwargs)
            return model
        
        assert action is not None and hp_bounds is not None,\
            "action and hp_bounds must both be set"
        model = cls._define_model_from_action(action, hp_bounds)
        return model
    
    @classmethod
    def _define_model_from_hp_config(cls, hp_config: HpConfig) -> Predictor:
        
        # Create a model instance by unpacking the HP dict
        model = cls.model_type(**hp_config.dict())
        
        return model
    
    @classmethod
    def _define_model_from_action(
            cls, 
            action: Iterable[float],
            hp_bounds: dict[str, tuple]
        ) -> Predictor:
        
        hp_config = cls.hp_config_type.from_action(
            action=action,
            bounds=hp_bounds
        )
        
        model = cls._define_model_from_hp_config(hp_config)
        
        return model
    
def define_family_type(
        name: str,
        model_type: type, 
        hp_config_type: type
    ) -> type:
    
    return type(
        name,
        (BaseFamily,),
        {
            "model_type": model_type,
            "hp_config_type": hp_config_type
        }
    )
