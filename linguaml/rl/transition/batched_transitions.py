from typing import Self
from pydantic import BaseModel, ConfigDict
from numpy import ndarray
import numpy as np

# Imports from this package
from ..state import State, BatchedStates
from ..action import Action, BatchedActions
from .transition import Transition

class BatchedTransitions(BaseModel):
    
    state: BatchedStates
    action: BatchedActions
    reward: ndarray
    advantage: ndarray
    log_prob: ndarray
    
    model_config = ConfigDict(
        arbitrary_types_allowed=True
    )
    
    @classmethod
    def from_transitions(cls, transitions: list[Transition]) -> Self:
        """Constructs a `BatchedTransitions` instance 
        from a list of `Transition` instances.
        
        Parameters
        ----------
        transitions : list[Transition]
            a list of `Transition` instances
            
        Returns
        -------
        Self
            a `BatchedTransitions` instance
        """
        
        states = []
        actions = []
        rewards = []
        advantages = []
        log_probs = []
        for transition in transitions:
            states.append(transition.state)
            actions.append(transition.action)
            rewards.append(transition.reward)
            advantages.append(transition.advantage)
            log_probs.append(transition.log_prob)
        
        return cls(
            state=BatchedStates.from_states(states),
            action=BatchedActions.from_actions(actions),
            reward=np.array(rewards, dtype=np.float32),
            advantage=np.array(advantages, dtype=np.float32),
            log_prob=np.array(log_probs, dtype=np.float32)
        )
