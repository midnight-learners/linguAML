from typing import Optional, Iterable
from enum import Enum
from ..hp import HpConfig
from .base import Model
from .svc import SVCFamily

class Family(Enum):
    
    SVC = SVCFamily
    
    @property
    def hp(self) -> type[HpConfig]:
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
    def n_hps(self) -> int:
        
        return self.value.n_hps()
    
    def define_model(
            self, 
            *,
            hp_config: Optional[HpConfig] = None,
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
