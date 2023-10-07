from abc import ABC, abstractmethod
from torch import Tensor
from torch import nn

class Agent(nn.Module, ABC):
    
    @abstractmethod
    def forward(self, state: Tensor) -> Tensor:
        pass
