from typing import Self, Optional
from enum import Enum
from torch import Tensor
from torch import nn
from torch.distributions import Distribution, Normal, Cauchy, Beta

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
            action_dim: int,
            distribution_family: DistributionFamily | str = DistributionFamily.Normal
        ) -> None:
        
        super().__init__()
        
        self._action_dim = action_dim
        if isinstance(distribution_family, str):
            distribution_family = DistributionFamily.from_name(distribution_family)
        self._distribution_family = distribution_family
        self._distribution_cls = distribution_family.value
        self._action = None
        self._distribution = None
        
        self.fc = nn.Linear(action_dim, 64)
        
        self.lstm = nn.LSTM(
            input_size=64,
            hidden_size=128,
            num_layers=1,
            batch_first=True
        )
        
        self.dist_param1 = nn.Sequential(
            nn.Linear(128, action_dim),
            nn.Sigmoid()
        )
        
        self.dist_param2 = nn.Sequential(
            nn.Linear(128, action_dim),
            nn.Sigmoid()
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
            - (state_dim, action_dim)
            - (N, state_dim, action_dim)

        Returns
        -------
        Distribution
            Shape:
            - (state_dim, action_dim)
            - (N, state_dim, action_dim)
        """
        
        x = self.fc(state)
        x, (h, c) = self.lstm(x)
        x = x[..., -1, :]
        
        
        # Generate the distribution
        distribution = self._distribution_cls(
            self.dist_param1(x), 
            self.dist_param2(x)
        )
        
        # Store the distribution
        self._distribution = distribution 
    
        return distribution
    
    def select_action(self, state: Optional[Tensor]) -> Tensor:
        
        # Genereate a new distribution
        # if the state is provided
        if state is not None:
            self.forward(state)
        
        # Select an action randomly from the distribution
        action = self._distribution.sample()
        
        # Clip the action by upper and lower bounds
        # More specifically, each entry of the vector must in between 0 and 1
        action = action.clip(0, 1)
        
        # Store the selected action
        self._action = action
        
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
        
        log_prob = self._distribution.log_prob(action).sum(dim=-1)
        
        return log_prob
