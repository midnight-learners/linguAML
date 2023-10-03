from typing import Optional, Iterable
from enum import Enum
from ..hp import HPConfig
from .base import Model
from .svc import SVCFamily

class Family(Enum):
    
    SVC = SVCFamily
    
    @property
    def hp(self) -> type[HPConfig]:
        """The class of the hyperparameter configuration
        of this model family.
        
        Notes
        -----
        It is a class insteat of an instance.
        """
        
        return self.value.hp()
    
    @property
    def param_names(self) -> tuple[str]:
        
        return self.value.hp().param_names()
    
    @property
    def numeric_param_names(self) -> tuple[str]:
        
        return self.value.hp().numeric_param_names()
    
    @property
    def categorical_param_names(self) -> tuple[str]:
        
        return self.value.hp().categorical_param_names()
    
    @property
    def n_hps(self) -> int:
        
        return self.value.n_hps()
    
    @property
    def n_numeric_hps(self) -> int:
        
        return len(self.numeric_param_names)
    
    @property
    def n_categorical_hps(self) -> int:
        
        return len(self.categorical_param_names)
    
    def n_levels_in_category(self, category: str) -> int:
        
        return self.value.hp().n_levels_in_category(category)
    
    def define_model(
            self, 
            *,
            hp_config: Optional[HPConfig] = None,
            action: Optional[Iterable[float]] = None,
            hp_bounds: Optional[dict[str, tuple]] = None,
            **kwargs
        ) -> Model:
        
        return self.value.define_model(
            hp_config=hp_config,
            action=action,
            hp_bounds=hp_bounds,
            **kwargs
        )
