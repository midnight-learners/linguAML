from typing import Self
from numpy import ndarray
import numpy as np
import torch
from torch import Tensor

# Imports from this package
from linguaml.tolearn.performance import PerformanceResult
from ..action import Action
from .config import StateConfig
from .state_unit import StateUnit

class State(StateConfig):
    
    def __init__(self, data: ndarray) -> None:
        
        # Check if the class attributes are set
        self.check_attributes() 
        
        self._data = data

    @property
    def data(self) -> ndarray:
        """Internal data representation of the state.
        It is a 2D NumPy array of shape (lookback, n_state_features).
        """
        
        return self._data
    
    @classmethod
    def from_state_units(cls, state_units: list[StateUnit]) -> Self:
        
        # Check that the number of state units is correct
        # It should be equal to the lookback
        assert len(state_units) == cls.lookback,\
            f"Expected {cls.lookback} state units, got {len(state_units)}"
            
        # Get the data from each state unit
        arrays = [state_unit.data for state_unit in state_units]
        
        # Stack the arrays
        data = np.stack(arrays)
        
        return cls(data)
    
    @classmethod
    def from_actions_and_rewards(cls, actions: list[Action], rewards: list[float]) -> Self:
        """Construct a state from a list of actions and rewards.

        Parameters
        ----------
        actions : list[Action]
            A list of actions.
        rewards : list[float]
            A list of rewards.

        Returns
        -------
        Self
            A state.
        """
            
        # Check that the number of actions is correct
        # It should be equal to the lookback
        assert len(actions) == cls.lookback,\
            f"Expected {cls.lookback} actions, got {len(actions)}"
        
        # Check that the number of rewards is correct
        # It should be equal to the lookback
        assert len(rewards) == cls.lookback,\
            f"Expected {cls.lookback} rewards, got {len(rewards)}"
            
        # Construct the state units
        state_units = [
            StateUnit.from_action_and_reward(action, reward)
            for action, reward in zip(actions, rewards)
        ]
    
        return cls.from_state_units(state_units)
    
    @classmethod
    def from_action_and_reward_pairs(cls, action_and_reward_pairs: list[tuple[Action, float]]) -> Self:
        """Construct a state from a list of action and reward pairs.

        Parameters
        ----------
        action_and_reward_pairs : list[tuple[Action, float]]
            A list of action and reward pairs.

        Returns
        -------
        Self
            A state.
        """
            
        # Check that the number of action and reward pairs is correct
        # It should be equal to the lookback
        assert len(action_and_reward_pairs) == cls.lookback,\
            f"Expected {cls.lookback} action and reward pairs, got {len(action_and_reward_pairs)}"
            
        # Construct the state units
        state_units = [
            StateUnit.from_action_and_reward(action, reward)
            for action, reward in action_and_reward_pairs
        ]
        
        return cls.from_state_units(state_units)
    
    @classmethod
    def from_performance_results(cls, performance_results: list[PerformanceResult]) -> Self:
        """Construct a state from a list of performance results.

        Parameters
        ----------
        performance_results : list[PerformanceResult]
            A list of performance results.

        Returns
        -------
        Self
            A state.
        """
        
        # Check that the number of performance results is correct
        # It should be equal to the lookback
        assert len(performance_results) == cls.lookback,\
            f"Expected {cls.lookback} performance results, got {len(performance_results)}"
            
        # Construct the state units
        state_units = [
            StateUnit.from_performance_result(performance_result)
            for performance_result in performance_results
        ]
        
        return cls.from_state_units(state_units)

    def to_tensor(self) -> Tensor:
        """Convert the state to a PyTorch tensor.

        Returns
        -------
        Tensor
            A PyTorch tensor.
        """
        
        return torch.tensor(self._data, dtype=torch.float32)
