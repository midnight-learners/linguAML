from typing import Optional
from torch import Tensor
from torch import nn
from torch.distributions import Distribution, Normal

class Agent(nn.Module):
    
    def __init__(
            self,
            action_dim: int
        ) -> None:
        
        super().__init__()
        
        self._action_dim = action_dim
        self._action = None
        self._distribution = None
        
        self._n_dist_params = 2
        
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
        """Action taken.
        """
        
        return self._action
        
    def forward(self, state: Tensor) -> Distribution:
        """_summary_

        Parameters
        ----------
        state : Tensor
            (state_dim, action_dim)
            (N, state_dim, action_dim)

        Returns
        -------
        Distribution
            _description_
        """
        
        x = self.fc(state)
        x, (h, c) = self.lstm(x)
        x = x[..., -1, :]
        
        # Extract parameters for the distribution
        mean = self.dist_param1(x)
        std = self.dist_param2(x)

        # Generate the distributino
        distribution = Normal(mean, std)
        
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
