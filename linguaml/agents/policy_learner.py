from typing import Optional
from torch import Tensor
from torch import nn
from torch.distributions import Distribution, Normal
from .base import Agent

class PolicyLearner(Agent):
    
    def __init__(self, action_dim: int) -> None:
        
        super().__init__()
        
        self.fc = nn.Linear(action_dim, 64)
        
        self.lstm_cell = nn.LSTMCell(
            input_size=64,
            hidden_size=128
        )
        
        self.dist_param1 = nn.Sequential(
            nn.Linear(128, action_dim),
            nn.Sigmoid()
        )
        
        self.dist_param2 = nn.Sequential(
            nn.Linear(128, action_dim),
            nn.Sigmoid()
        )
    
    def forward(self, state: Tensor) -> Tensor:
        
        return super().forward(state)