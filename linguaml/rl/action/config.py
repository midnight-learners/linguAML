from abc import ABC
from linguaml.tolearn.family import Family
from linguaml.tolearn.hp.bounds import NumericHPBounds

class ActionConfig(ABC):
    """The abstract class of the action configuration.
    It contains the following class attributes:
    - family: Family
    - numeric_hp_bounds: NumericHPBounds
    """
    
    family: Family
    numeric_hp_bounds: NumericHPBounds
    
    @classmethod
    def check_attributes(cls) -> None:
        """Check if the class attributes are set.
        """
        
        assert hasattr(cls, "family"),\
            "The class attribute 'family' must be set."
        assert hasattr(cls, "numeric_hp_bounds"),\
            "The class attribute 'numeric_hp_bounds' must be set."
