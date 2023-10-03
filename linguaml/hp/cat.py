from enum import StrEnum
import numpy as np

class CategoricalHP(StrEnum):
    
    def __str__(self) -> str:
        return self.value
    
    def __repr__(self) -> str:
        return f"'{str(self)}'"
    
    @classmethod
    def n_levels(cls) -> int:
        """Total number of levels in this category.
        """
        
        return len(cls)
    
    @property
    def level_index(self) -> int:
        """The index of the level in this category.
        """
        
        return tuple(self.__class__).index(self)
    
    @property
    def one_hot(self) -> np.ndarray:
        """The one-hot encoding vector of this level.
        """

        I = np.eye(self.n_levels())
        
        return I[self.level_index]
