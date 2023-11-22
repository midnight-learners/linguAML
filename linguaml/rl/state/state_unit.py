from typing import Self
from numpy import ndarray
import numpy as np

# Imports from this package
from linguaml.tolearn.hp import CategoricalHP
from linguaml.tolearn.performance import PerformanceResult
from linguaml.rl.action import Action
from linguaml.rl.state.config import StateConfig

class StateUnit(StateConfig):
    
    def __init__(self, data: ndarray) -> None:
        
        self._data = data
    
    @property
    def data(self) -> ndarray:
        """Internal data of the state unit. 
        This is a numpy array.
        """
        
        return self._data
        
    @classmethod
    def from_action_and_reward(cls, action: Action, reward: float) -> Self:
        
        # Continuous part
        entries = []
        for hp_name in action.family.hp().numeric_hp_names():
            entry = action[hp_name]
            entries.append(entry)
        continuous_part = np.array(entries)
        
        # Discrete part
        one_hot_arrays = []
        for hp_name in action.family.hp().categorical_hp_names():
            # Level index of the categorical hyperparameter
            level_index = action[hp_name]
            
            # Get the class of the categorical hyperparameter
            categorical_hp_type: type[CategoricalHP] = action.family.hp().hp_type(hp_name)
            
            # Recover the categorical hyperparameter
            hp = categorical_hp_type.from_index(level_index)
           
           # Collect the one-hot array
            one_hot_arrays.append(hp.one_hot)
        
        # Create the array of the discrete part
        discrete_part = np.concatenate(one_hot_arrays)
        
        # Reward part
        # The reward part is a singleton array containing the reward
        reward_part = np.array([reward])
        
        # Construct the final encoded array
        encoded_array = np.concatenate([continuous_part, discrete_part, reward_part])
        
        return cls(encoded_array)
            
    @classmethod
    def from_performance_result(cls, performance_result: PerformanceResult) -> Self:
        
        # Get the hyperparameter configuration
        hp_config = performance_result.hp_config
        
        # The reward is the score
        reward = performance_result.score
        
        # Convert to action
        action = Action.from_hp_config(hp_config)
                
        return cls.from_action_and_reward(action, reward)
