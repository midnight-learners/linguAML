from typing import Optional, Iterable
from enum import Enum
import inspect
from ..hp import HPConfig
from .base import Family as BaseFamily
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

def _set_methods():

    method_names = tuple(map(
        lambda item: item[0],
        inspect.getmembers(BaseFamily, predicate=inspect.ismethod)
    ))

    for method_name in method_names:
        
        def new_method(self, *args, **kwargs):
            
            # Get the class method of the hyperparameter configuration type
            # The class type itself can be accessed via `self.value.hp_config_type`
            method = getattr(self.value.hp_config_type, method_name)

            return method(*args, **kwargs)
        
        # Set the method of Family enum
        setattr(Family, method_name, new_method)
        
_set_methods()
