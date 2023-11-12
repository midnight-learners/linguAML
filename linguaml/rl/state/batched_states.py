
from numpy import ndarray
import torch
from torch import Tensor

class BatchedStates:
    
    def __init__(self, data: ndarray) -> None:
        
        self._data = data
    
    @property
    def shape(self) -> tuple[int]:
        
        return self._data.shape
    
    def to_tensor(self) -> Tensor:
        
        return torch.tensor(self._data, dtype=torch.float32)
    