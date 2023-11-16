from pydantic import BaseModel, ConfigDict
import torch
from torch import Tensor

# Imports from this package
from ..state import State
from ..action import Action

class Transition(BaseModel):
    
    state: State
    action: Action
    reward: float
    advantage: float
    log_prob: float
    
    model_config = ConfigDict(
        arbitrary_types_allowed=True
    )
