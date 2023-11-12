from typing import Self, Optional
from enum import Enum

import torch
from torch import Tensor
from torch import nn
from torch.nn import functional as F
from torch.distributions import Distribution, Normal, Cauchy, Beta, Categorical

# Imports from this package
from linguaml.tolearn.hp import CategoricalHP
from linguaml.tolearn.families.base import Family
from .state import State, BatchedStates, calc_n_state_features
from .action import Action, BatchedActions

class ContinuousDistributionFamily(Enum):
    
    NORMAL = Normal
    CAUCHY = Cauchy
    BETA = Beta
    
    @classmethod
    def from_name(cls, name: str) -> Self:
        """Specify the distribution family by the name.

        Parameters
        ----------
        name : str
            - "normal" or "gaussian"
            - "cauchy"
            - "beta"

        Returns
        -------
        Self
            A variant of this enumeration.
        """
        
        # Convert to lower case
        name = name.lower()
        
        match name:
            case "normal" | "guassian":
                return ContinuousDistributionFamily.NORMAL
            case "cauchy":
                return ContinuousDistributionFamily.CAUCHY
            case "beta":
                return ContinuousDistributionFamily.BETA
            
class Agent(nn.Module):
    
    n_cont_dist_params = 2
    
    def __init__(
            self,
            family: Family,
            hidden_size: int = 128,
            cont_dist_family: ContinuousDistributionFamily | str = ContinuousDistributionFamily.NORMAL
        ) -> None:
        
        super().__init__()
        
        self._family = family
        
        # Number of state features, i.e., the input features
        self._n_state_features = calc_n_state_features(family)
        
        if isinstance(cont_dist_family, str):
            cont_dist_family = ContinuousDistributionFamily.from_name(cont_dist_family)
        self._cont_dist_family = cont_dist_family
        self._cont_dist_cls = cont_dist_family.value
        
        self._action = None
        self._distributions: dict[str, Distribution] = {}
        
        # Layers
        
        self.fc = nn.Linear(self._n_state_features, 64)
        
        self.lstm = nn.LSTM(
            input_size=64,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True
        )
        
        # Continuous
        
        for hp_name in self._family.numeric_hp_names():
            
            # The layer to generate the first parameter of the distribution
            cont_dist_param1 = nn.Sequential(
                nn.Linear(hidden_size, 1),
                nn.Sigmoid()
            )
            
            # The layer to generate the second parameter of the distribution
            cont_dist_param2 = nn.Sequential(
                nn.Linear(hidden_size, 1),
                nn.Sigmoid()
            )
            
            # This layer will generate the parameters of 
            # the continuous distribution for the numeric hyperparameter
            layer = nn.ModuleDict({
                "param1": cont_dist_param1,
                "param2": cont_dist_param2
            })
            
            # Add the layer
            # It is named by the numeric hyperparameter name
            self.add_module(
                hp_name,
                layer
            )
        
        # Discrete
        
        for hp_name in self._family.categorical_hp_names():
            
            # Name of the category type
            category_type: type[CategoricalHP] = self._family.hp_type(hp_name)
            
            # Number of levels in the category
            n_levels = category_type.n_levels()
            
            # This layer will assign probability 
            # to each level of the category
            layer = nn.Sequential(
                nn.Linear(hidden_size, n_levels),
                nn.Softmax(dim=-1)
            )
            
            # Add the layer
            # It is named by the category hyperparameter name
            self.add_module(
                hp_name,
                layer
            )
  
    @property
    def distributions(self) -> dict[str, Distribution]:
        """Distributions to generate the action.
        """
        
        return self._distributions
    
    @property
    def action(self) -> Tensor:
        """The latest taken action.
        """
        
        # If there are several actions,
        # then return the last one
        # This is caused by passing a batch of states
        if len(self._action.shape) > 1:
            action = self._action[-1]
            
        else:
            action = self._action
        
        # Detach from the computation graph
        action = action.detach()
        
        return action
    
    def forward(self, state: State | BatchedStates) -> dict[str, Distribution]:
        """Generate distributions to sample an action from.

        Parameters
        ----------
        
        """
        
        input_tensor = state.to_tensor()
        
        x = self.fc(input_tensor)
        x, (h, c) = self.lstm(x)
        x = x[..., -1, :]
        
        # Create a dictionary to store distributions
        distributions = {}
        
        # Generate continuous distributions
        # for numeric hyperparameters
        for hp_name in self._family.numeric_hp_names():
                
            # Get the layer module to generate parameters
            layer = self.get_submodule(hp_name)
            
            # Generate the parameters
            param1 = layer["param1"](x)
            param2 = layer["param2"](x)
            
            # Generate the distribution
            dist = self._cont_dist_cls(param1, param2)
            
            # Set the distribution for this numeric hyperparameter
            distributions[hp_name] = dist
        
        # Generate discrete distributions
        # for categorical hyperparameters
        for hp_name in self._family.categorical_hp_names():
            
            # Get the layer module to generate probability for each level
            category_layer = self.get_submodule(hp_name)
            
            # Ptrobabilites of all category levels
            probs = category_layer(x)
            
            # Generate the discrete distribution
            dist = Categorical(probs)
            
            # Set the distribution for this category
            distributions[hp_name] = dist
        
        # Store the distribution
        self._distributions = distributions
    
        return distributions
    
    def select_action(
            self, 
            state: Optional[State | BatchedStates] = None
        ) -> Action | BatchedActions:
        
        # Generate a new distribution
        # if the state is provided
        if state is not None:
            self.forward(state)
            
        # Create an empty action or batched actions
        if isinstance(state, State):
            action = Action()
        elif isinstance(state, BatchedStates):
            action = BatchedActions()
        else:
            raise TypeError(f"Unsupported type of input state: {type(state)}")
        
        # Sample a value for each hyperparameter
        for hp_name in self._family.hp_names():
            
            # Get the distribution
            dist = self._distributions[hp_name]
            
            # Sample a value
            value = dist.sample()
            
            # Clip the value if it is a continuous value
            # and it is out of the range [0, 1]
            if hp_name in self._family.numeric_hp_names():
                value = torch.clip(value, 0, 1)
            
            # Store the value
            action[hp_name] = value

        # Store the selected action
        self._action = action
        
        return action
     