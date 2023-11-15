from typing import Self
from enum import StrEnum
from numpy import ndarray
import numpy as np

class CategoricalHP(StrEnum):
    
    def __str__(self) -> str:
        return self.value
    
    def __repr__(self) -> str:
        return f"'{str(self)}'"
    
    @classmethod
    def level_values(cls) -> tuple[str]:
        """Values, i.e., string representations, of all levels.
        """
        
        return tuple(cls)
    
    @classmethod
    def from_index(cls, idx: int) -> Self:
        """Returns the level of this category 
        based on the given level index.

        Parameters
        ----------
        idx : int
            Level index.

        Returns
        -------
        Self
            Level.
        """
        
        return tuple(cls.__members__.values())[idx]
    
    @classmethod
    def from_one_hot(cls, one_hot: ndarray) -> Self:
        """Constructs a member from the given one-hot encoding vector.
        
        Parameters
        ----------
        one_hot : ndarray
            One-hot encoding vector.

        Returns
        -------
        Self
            Member.
        """
        
        return cls.from_index(np.argmax(one_hot))
    
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
    def one_hot(self) -> ndarray:
        """The one-hot encoding vector of this level.
        """

        I = np.eye(self.n_levels())
        
        return I[self.level_index]
