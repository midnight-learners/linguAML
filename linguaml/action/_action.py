from typing import Self
import numpy as np
import torch
from torch import Tensor

# Imports from this package
from linguaml.families.base import Family
from linguaml.hp import HPConfig, CategoricalHP
from linguaml.hp.bounds import NumericHPBounds

class Action:
    
    family: Family
    numeric_hp_bounds: NumericHPBounds
    
    def __init__(self, data: np.ndarray) -> None:
        """Construct an action from a NumPy array.

        Parameters
        ----------
        data : np.ndarray
            The action represented as a NumPy array.
        """
        
        self._data = data
        
        # Validate the class attributes
        assert hasattr(self.__class__, "family"),\
            "Class attribute 'family' must be set."
        assert hasattr(self.__class__, "numeric_hp_bounds"),\
            "Class attribute 'numeric_hp_bounds' must be set."
        
        # Validate the class attribute numeric_hp_bounds
        if not isinstance(self.__class__.numeric_hp_bounds, NumericHPBounds):
            self.__class__.numeric_hp_bounds = NumericHPBounds.model_validate(self.__class__.numeric_hp_bounds)
    
    @property
    def continuous(self) -> np.ndarray:
        """The continuous part of the action.
        """
        
        return self._data[..., :self.family.n_numeric_hps()]
    
    @property
    def discrete(self) -> dict[str, int]:
        """The discrete part of the action.
        """

        # The start index of the discrete part of the action
        start = self.family.n_numeric_hps()
        
        # Dictionary of discrete actions
        discrete_actions = {}
        
        # For each categorical hyperparameter, extract the 
        for name in self.family.categorical_hp_names():
        
            # The end associated with the current category
            end = start + self.family.n_levels_in_category(name)
    
            # One-hot encoding vector representing 
            # the choice of the category level
            one_hot_action = self._data[..., start:end]
            
            # Discrete action, i.e., the level index
            disc_action = one_hot_action.argmax(axis=-1)
            
            # Add the discrete action to the dictionary
            discrete_actions[name] = disc_action
            
            # Update the start index
            start += self.family.n_levels_in_category(name)
        
        return discrete_actions
    
    @classmethod
    def from_hp_config(cls, hp_config: HPConfig) -> Self:
        """Construct an action from a hyperparameter configuration.

        Parameters
        ----------
        hp_config : HPConfig
            The hyperparameter configuration.

        Returns
        -------
        Self
            The action.
        """
        
        # Get continuous action
        cont_action_entries = []
        for hp_name in hp_config.numeric_hp_names():
            hp_value = getattr(hp_config, hp_name)
            hp_max = cls.numeric_hp_bounds.name2bounds[hp_name].max
            hp_min = cls.numeric_hp_bounds.name2bounds[hp_name].min
            action_value = (hp_value - hp_min) / (hp_max - hp_min)
            cont_action_entries.append(action_value)
        cont_action = np.array(cont_action_entries)
        
        # Get discrete actions
        disc_actions = []
        for hp_name in hp_config.categorical_hp_names():
            hp_value: CategoricalHP = getattr(hp_config, hp_name)
            disc_actions.append(hp_value.one_hot)
        disc_actions = np.concatenate(disc_actions)
        
        # Generate the final action
        data = np.concatenate([cont_action, disc_actions])
        
        return cls(data)
    
    @classmethod
    def from_tensor(cls, tensor: Tensor) -> Self:
        """Construct an action from a PyTorch tensor.

        Parameters
        ----------
        action : Tensor
            The PyTorch tensor representing the action.

        Returns
        -------
        Self
            The action.
        """
        
        return cls(tensor.detach().numpy())
    
    def to_hp_config(self) -> HPConfig:
        """Convert the action to a hyperparameter configuration.

        Returns
        -------
        HPConfig
            The hyperparameter configuration corresponding to the action.
        """
        
        # Dictionary of hyperparameters
        hps = {}
        
        # Iterate over all slots in the action
        for i, hp in enumerate(self.continuous):
            
            # Find all numberic hyperparameters
            hp_name = self.family.numeric_hp_names()[i]
            
            # Get the HP type, which is either float or int
            hp_type = self.family.hp_type(hp_name)
            
            # Lower and upper bounds
            bounds = self.numeric_hp_bounds.get_bounds(hp_name)
            
            # Recover the hyperparameter value
            hp = hp_type(hp * (bounds.max - bounds.min) + bounds.min)
            
            hps[hp_name] = hp
            
        # Extract the discrete actions
        for hp_name in self.family.categorical_hp_names():
            
            # The index of the level of the category
            disc_action: int = self.discrete[hp_name]
            
            # Categorical hyperparameter value
            categorical_hp_type = self.family.hp_type(hp_name)
            hp = categorical_hp_type.from_index(disc_action)
            
            hps[hp_name] = hp
        
        # Create the instances by unpacking hps
        hp_config = self.family.hp()(**hps)
        
        return hp_config
    
    def to_tensor(self) -> Tensor:
        """Convert the action to a PyTorch tensor.

        Returns
        -------
        Tensor
            The PyTorch tensor representing the action.
        """
        
        return torch.from_numpy(self._data)
