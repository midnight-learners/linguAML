from typing import Self, Optional
from enum import Enum
import inflection

import torch
from torch import Tensor
from torch import nn
from torch.nn import functional as F
from torch.distributions import Distribution, Normal, Cauchy, Beta, Categorical

from .utils import (
    calc_action_dim,
    extract_cont_action,
    extract_disc_action
)
from .hp import CategoricalHP
from .families.base import Family


class DistributionFamily(Enum):
    
    Normal = Normal
    Cauchy = Cauchy
    Beta = Beta
    
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
                return DistributionFamily.Normal
            case "cauchy":
                return DistributionFamily.Cauchy
            case "beta":
                return DistributionFamily.Beta
            
class Agent(nn.Module):
    
    n_dist_params = 2
    
    def __init__(
            self,
            family: Family,
            hidden_size: int = 128,
            cont_dist_family: DistributionFamily | str = DistributionFamily.Normal
        ) -> None:
        
        super().__init__()
        
        self._family = family
        self._action_dim = calc_action_dim(family)
        
        if isinstance(cont_dist_family, str):
            cont_dist_family = DistributionFamily.from_name(cont_dist_family)
        self._cont_dist_family = cont_dist_family
        self._cont_dist_cls = cont_dist_family.value
        
        self._action = None
        self._distribution: dict[str, Distribution] = {}
        
        # Layers
        
        self.fc = nn.Linear(self._action_dim, 64)
        
        self.lstm = nn.LSTM(
            input_size=64,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True
        )
        
        # Continuous
        
        self.cont_dist_param1 = nn.Sequential(
            nn.Linear(hidden_size, self._family.n_numeric_hps()),
            nn.Sigmoid()
        )
        
        self.cont_dist_param2 = nn.Sequential(
            nn.Linear(hidden_size, self._family.n_numeric_hps()),
            nn.Sigmoid()
        )
        
        # Discrete
        
        for category in self._family.categorical_hp_names():
            
            # Name of the category type
            category_type: type[CategoricalHP] = self._family.hp_type(category)
            
            # Number of levels in the category
            n_levels = category_type.n_levels()
            
            # This layer will assign probability 
            # to each level of the category
            layer = nn.Sequential(
                nn.Linear(hidden_size, n_levels),
                nn.Softmax(dim=-1)
            )
            
            # Create a name for this layer
            # It is named by the category name
            layer_name = inflection.underscore(category)
            
            # Add the layer
            self.add_module(
                layer_name,
                layer
            )
  
    @property
    def distribution(self) -> Distribution:
        """Distribution to generate the action.
        """
        
        return self._distribution
    
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
        
    def forward(self, state: Tensor) -> Distribution:
        """_summary_

        Parameters
        ----------
        state : Tensor
            Shape:
            - (n_configs, n_hps)
            - (N, n_configs, n_hps)

        Returns
        -------
        Distribution
            Shape:
            - (n_configs, n_hps)
            - (N, state_dim, n_hps)
        """
        
        x = self.fc(state)
        x, (h, c) = self.lstm(x)
        x = x[..., -1, :]
        
        distribution = {}
        
        # Generate a continuous distribution
        # for continuous hyperparameters
        cont_dist = self._cont_dist_cls(
            self.cont_dist_param1(x), 
            self.cont_dist_param2(x)
        )
        distribution["continuous"] = cont_dist
        
        # Generate discrete distributions
        # for discrete hyperparameters
        for name in self._family.categorical_hp_names():
            
            # Get the layer module to generate probability for each level
            category_layer = self.get_submodule(name)
            
            # Ptrobabilites of all category levels
            probs = category_layer(x)
            
            # Generate the discrete distribution
            dist = Categorical(probs)
            
            # Set the distribution for this category
            distribution[name] = dist
        
        # Store the distribution
        self._distribution = distribution 
    
        return distribution
    
    def select_action(
            self, 
            state: Optional[Tensor] = None
        ) -> Tensor:
        
        # Generate a new distribution
        # if the state is provided
        if state is not None:
            self.forward(state)
        
        # Continuous action
        cont_action = self._select_cont_action()
        
        # All discrete actions
        disc_actions = [
            self._select_disc_action(category)
            for category in self._family.categorical_hp_names()
        ]
        
        # Concatenate the continuous and discete actions
        action = torch.concat((cont_action, *disc_actions), dim=-1)
        
        # Store the selected action
        self._action = action
        
        return action
        
    def _select_cont_action(self) -> Tensor:
        
        # Select an action randomly from the distribution
        action = self._distribution["continuous"].sample()
        
        # Clip the action by upper and lower bounds
        # More specifically, each entry of the vector must in between 0 and 1
        action = action.clip(0, 1)
        
        return action
    
    def _select_disc_action(self, category: str) -> Tensor:
           
        # Select an action randomly from the distribution, i.e.,
        # the index of the category level
        action = self._distribution[category].sample()
        
        # Convert the integer value of the action
        # to a vector using one-hot encoding
        action = F.one_hot(
            action,
            num_classes=self._family.n_levels_in_category(category)
        ).type(torch.float)
        
        return action

    def log_prob(
            self, 
            action: Optional[Tensor] = None, 
            state: Optional[Tensor] = None
        ) -> Tensor:
        """Compute the log-probability of the action taken.

        Parameters
        ----------
        action : Optional[Tensor], optional
            _description_, by default None
        state : Optional[Tensor], optional
            _description_, by default None

        Returns
        -------
        Tensor
            _description_
        """
        
        if action is None:
            assert state is None,\
                "state must be set None since action is None"
            action = self._action
        
        if state is not None:
            self.forward(state)
            
        log_prob_of_cont_action = self._log_prob_of_cont_action(action)  
        
        log_prob_of_disc_actions = [
            self._log_prob_of_disc_action(categorical_hp_name, action)
            for categorical_hp_name in self._family.categorical_hp_names()
        ]
        
        log_prob = torch.sum(
            torch.stack((
                log_prob_of_cont_action,
                *log_prob_of_disc_actions
            )),
            dim=0
        )
        
        return log_prob
    
    def _log_prob_of_cont_action(self, action: Optional[Tensor] = None) -> Tensor:
        
        if action is None:
            action = self._action
            
        cont_action = extract_cont_action(
            action=action,
            family=self._family
        )
        
        log_prob = self._distribution["continuous"].log_prob(cont_action).sum(dim=-1)
        
        return log_prob

    def _log_prob_of_disc_action(
            self, 
            categorical_hp_name: str, 
            action: Optional[Tensor] = None
        ) -> Tensor:
        
        if action is None:
            action = self._action
            
        disc_action = extract_disc_action(
            action=action,
            family=self._family,
            categorical_hp_name=categorical_hp_name
        )
        
        log_prob = self._distribution[categorical_hp_name].log_prob(disc_action)
        
        return log_prob
