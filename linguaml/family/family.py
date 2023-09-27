from typing import Optional, Iterable
from enum import Enum
from ..hp import HpConfig
from .base import Predictor
from .svc import SVCFamily

class Family(Enum):
    
    SVC = SVCFamily
    
    @property
    def hp(self) -> type:
        
        return self.value.hp()
    
    @property
    def n_hps(self) -> int:
        
        return self.value.n_hps()
    
    def define_model(
            self, 
            hp_config: Optional[HpConfig] = None,
            *,
            action: Optional[Iterable[float]] = None,
            hp_bounds: Optional[dict[str, tuple]] = None,
            **kwargs
        ) -> Predictor:
        
        return self.value.define_model(
            hp_config,
            action=action,
            hp_bounds=hp_bounds,
            **kwargs
        )
