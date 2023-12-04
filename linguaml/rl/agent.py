from typing import Self, Optional
from enum import Enum
import numpy as np

import torch
from torch import Tensor
from torch import nn
from torch.distributions import Distribution, Normal, Cauchy, Beta, Categorical

# Imports from this package
from linguaml.tolearn.hp import CategoricalHP
from linguaml.tolearn.hp.bounds import NumericHPBounds
from linguaml.tolearn.family import Family
from .state import State, BatchedStates, calc_n_state_features
from .action import ActionConfig, Action, BatchedActions

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
            A member of this enumeration.
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
            numeric_hp_bounds: NumericHPBounds | dict,
            hidden_size: int = 128,
            cont_dist_family: ContinuousDistributionFamily | str = ContinuousDistributionFamily.NORMAL
        ) -> None:
        
        super().__init__()
        
        # Family of the model to fine tune
        self._family = family
        
        # Bounds of numeric hyperparameters
        if not isinstance(numeric_hp_bounds, NumericHPBounds):
            numeric_hp_bounds = NumericHPBounds.model_validate(numeric_hp_bounds)
        numeric_hp_bounds = numeric_hp_bounds
        
        # Set up action configuration
        ActionConfig.family = family
        ActionConfig.numeric_hp_bounds = numeric_hp_bounds

        # Number of state features, i.e., the input features
        self._n_state_features = calc_n_state_features(family)
        
        if isinstance(cont_dist_family, str):
            cont_dist_family = ContinuousDistributionFamily.from_name(cont_dist_family)
        self._cont_dist_family = cont_dist_family
        self._cont_dist_cls = cont_dist_family.value
        
        # Distributions to generate the action
        self._distributions: dict[str, Distribution] = {}
        
        # Whether the last state is batched
        self._is_last_state_batched: bool = False
        
        ##############################
        # Layers
        ##############################
        
        self.fc = nn.Sequential(
            nn.Linear(self._n_state_features, 64),
            nn.ReLU(inplace=True),
        )
        
        self.lstm = nn.LSTM(
            input_size=64,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True
        )
        
        # Layers to generate parameters of continuous distributions
        # for numeric hyperparameters
        
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
        
        # Layers to generate probabilities of categorical distributions
        # for categorical hyperparameters
        
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
    def family(self) -> Family:
        """Model family to learn.
        """
        
        return self._family
    
    def forward(self, state: State | BatchedStates) -> dict[str, Distribution]:
        """Forward pass. Generate distributions for all hyperparameters.

        Parameters
        ----------
        state : State | BatchedStates
            State or batched states.

        Returns
        -------
        dict[str, Distribution]
            A dictionary mapping hyperparameter names to distributions.
        """
        
        # Convert input state to tensor
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
            # - If the input is a single state, each parameter has shape (1,)
            # - If the input is a batch of states, each parameter has shape (N, 1)
            param1: Tensor = layer["param1"](x)
            param2: Tensor = layer["param2"](x)
            
            # Flatten the tensors
            param1 = param1.flatten()
            param2 = param2.flatten()
            
            # If the length of the tensor is 1,
            # then convert it to a scalar
            if len(param1) == 1:
                param1 = param1[0]
            if len(param2) == 1:
                param2 = param2[0]
            
            # Generate the distribution
            distribution = self._cont_dist_cls(param1, param2)
            
            # Set the distribution for this numeric hyperparameter
            distributions[hp_name] = distribution
        
        # Generate discrete distributions
        # for categorical hyperparameters
        for hp_name in self._family.categorical_hp_names():
            
            # Get the layer module to generate probability for each level
            category_layer = self.get_submodule(hp_name)
            
            # Ptrobabilites of all category levels
            probs = category_layer(x)
            
            # Generate the discrete distribution
            distribution = Categorical(probs)
            
            # Set the distribution for this category
            distributions[hp_name] = distribution
        
        # Store the distribution
        self._distributions = distributions
        
        # Update the flag to indicate whether the last state is batched
        self._is_last_state_batched = isinstance(state, BatchedStates)
    
        return distributions
    
    def select_action(
            self, 
            state: State | BatchedStates
        ) -> Action | BatchedActions:
        """Select an action based on the state.
        - If the state is a single state, then the action is a single action.
        - If the state is a batch of states, then the action is a batch of actions.

        Parameters
        ----------
        state : State | BatchedStates
            State or batched states.

        Returns
        -------
        Action | BatchedActions
            An action or batched actions.
        """
        
        # Generate distributions
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
            distribution = self._distributions[hp_name]
            
            # Sample a value
            value = distribution.sample()
            
            # Clip the value if it is a continuous value
            # and it is out of the range [0, 1]
            if hp_name in self._family.numeric_hp_names():
                value = torch.clip(value, 0, 1)
            
            # Set the action value
            action[hp_name] = value
        
        return action
    
    def select_random_action(self) -> Action:
        """Select a random action.

        Returns
        -------
        Action
            A random action.
        """
        
        # Create an empty action
        action = Action()
        
        # Continuous actions
        for hp_name in self._family.numeric_hp_names():
                
            # Generate a random number in [0, 1]
            action[hp_name] = np.random.rand()
            
        # Discrete actions
        for hp_name in self._family.categorical_hp_names():
            
            # Get the number of levels in the category
            n_levels = self._family.n_levels_in_category(hp_name)
            
            # Generate a random integer in [0, n_levels - 1]
            action[hp_name] = np.random.randint(0, n_levels)
        
        return action
     
    def log_prob(
            self, 
            action: Action | BatchedActions, 
            state: Optional[State | BatchedStates] = None
        ) -> Tensor:
        """Compute the log-probability of the action taken.
        If the state is provided, then the log-probability is computed based on the state.
        Otherwise, the log-probability is computed based on the last state.

        Parameters
        ----------
        action : Action | BatchedActions
            Action or batched actions.
        state : Optional[State  |  BatchedStates], optional
            State or batched states, by default None.

        Returns
        -------
        Tensor
            Log-probability of the action.
            - If the action is a single action, then the log-probability is a single value.
            - If the action is a batch of actions, then the log-probability is a batch of values.
        """
        
        # Generate distributions if the state is provided
        if state is not None:
            self.forward(state)
        
        # Ensure that the action and state are compatible
        if self._is_last_state_batched:
            assert isinstance(action, BatchedActions),\
                "The action should be batched since the last state is batched."
        else:
            assert isinstance(action, Action),\
                "The action should be a single action since the last state is a single state."
        
        # Compute the log probability of the action
        log_probs = []
        for hp_name in self._family.hp_names():
            
            # Get the distribution
            distribution = self._distributions[hp_name]
            
            # Compute the log probability of the action
            log_prob = distribution.log_prob(action.to_tensor_dict()[hp_name])
            
            # Store the log probability of single action
            log_probs.append(log_prob)
        
        # Sum the log probabilities of all actions
        log_prob = torch.sum(torch.stack(log_probs), dim=0)
        
        return log_prob
