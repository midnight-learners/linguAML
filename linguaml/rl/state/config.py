from abc import ABC

class StateConfig(ABC):
    """The abstract class of the state configuration.
    It contains the following class attributes:
    - lookback: int
        Number of hyperparameter configurations 
        the agent will look back before selecting an action.
    """
    
    lookback: int

    @classmethod
    def check_attributes(cls) -> None:
        """Check if the class attributes are set.
        """
        
        assert hasattr(cls, "lookback"),\
            "The class attribute 'lookback' is not set"
