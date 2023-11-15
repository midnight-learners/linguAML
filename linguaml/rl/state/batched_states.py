from typing import Self
from numpy import ndarray
import numpy as np
import torch
from torch import Tensor

# Imports from this package
from .config import StateConfig
from .state import State

class BatchedStates(StateConfig):
    
    def __init__(self, data: ndarray) -> None:
        
        # Check if the class attributes are set
        self.check_attributes() 
        
        # Check that data is of the right shape
        assert len(data.shape) == 3,\
            f"Data must be of shape (batch_size, {self.lookback}, n_state_features)"
        assert data.shape[1] == self.lookback,\
            f"Data must be of shape (batch_size, {self.lookback}, n_state_features)"
        
        self._data = data
        
    @property
    def data(self) -> ndarray:
        """Internal data of the state.
        It is of shape (batch_size, lookback, n_state_features).
        """
        
        return self._data
        
    @classmethod
    def from_states(cls, states: list[State]) -> Self:
        
        # Get the data from each state
        arrays = [state.data for state in states]
        
        # Stack the arrays
        data = np.stack(arrays)
        
        return cls(data)
    
    def to_tensor(self) -> Tensor:
        """Converts the data to a PyTorch tensor.
        
        Returns
        -------
        Tensor
            A PyTorch tensor.
        """
        
        return torch.from_numpy(self.data).float()
