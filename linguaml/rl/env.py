from typing import Optional, Iterable
from collections import deque
import multiprocessing
from numpy import ndarray
import numpy as np
from sklearn.metrics import accuracy_score

# Imports from this package
from linguaml.types import Model
from linguaml.data.dataset import Dataset
from .action import Action
from .state import StateConfig, State

class Env:
    
    def __init__(
            self,
            datasets: list[Dataset],
            *,
            lookback: int = 10,
            fitting_time_limit: float = 5.0,
            random_state: Optional[int] = None,
        ) -> None:
        
        # Set up state configuation
        StateConfig.lookback = lookback
        
        # Time limit of fitting the model
        self._fitting_time_limit = fitting_time_limit

        # Random state
        self._random_state = random_state
        
        # A NumPy random generator
        self._rng = np.random.default_rng(random_state)
        
        # Dataset
        self._datasets = datasets
        
        # A buffer of actions and associated rewards
        self._action_reward_buffer = deque(maxlen=lookback)
        
        # Reset env
        self._initial_state = self.reset()
        
    @property
    def datasests(self) -> list[Dataset]:
        """Datesets to work on.
        """
        
        return self._datasets
    
    @property
    def initial_state(self) -> State:
        """Initial state.
        """
        
        return self._initial_state

    def reset(self) -> State:
        """Reset the environment and return an initial state.

        Returns
        -------
        State
            An initial state.
        """
        
        # Clear the buffer of actions and rewards
        self._action_reward_buffer.clear()
        
        for _ in range(StateConfig.lookback):
            
            # Randomly choose a dataset
            dataset: Dataset = self._rng.choice(self._datasets)
            
            # Generate a random action
            action = Action.random(random_state=self._random_state)
            
            # Convert action to a hyperparameter configuration
            hp_config = action.to_hp_config()
            
            # Difine the model with the hyperparameter configuation
            model = action.family.define_model(
                hp_config=hp_config
            )
            
            # Get reward by fitting the model in a separate process
            reward = self._fit_model(model, dataset)
            
            # Put the action and reward into the buffer
            self._action_reward_buffer.append((
                action, 
                reward if reward is not None else 0.0
            ))
        
        # Create an initial state
        initial_state = State.from_action_and_reward_pairs(
            list(self._action_reward_buffer)
        )
        self._initial_state = initial_state
        
        return initial_state
    
    def step(self, action: Action) -> tuple[State, float | None]:
        """Take an action and return the next state and reward.

        Parameters
        ----------
        action : Action
            Action taken by the agent.

        Returns
        -------
        tuple[State, float | None]
            Next state and reward.
            If the reward is None, it means that the fitting time limit is exceeded.
        """
        
        # Randomly choose a dataset
        dataset: Dataset = self._rng.choice(self._datasets)
        
        # Convert action to a hyperparameter configuration
        hp_config = action.to_hp_config()
        
        # Difine the model with the hyperparameter configuation
        model = action.family.define_model(
            hp_config=hp_config
        )
        
        # Get reward by fitting the model in a separate process
        reward = self._fit_model(model, dataset)
        
        # Put the action and reward into the buffer
        self._action_reward_buffer.append((
            action, 
            reward if reward is not None else 0.0
        ))
        
        # Generate the next state
        state = State.from_action_and_reward_pairs(
            list(self._action_reward_buffer)
        )
        
        return state, reward
    
    def _fit_model(self, model: Model, dataset: Dataset) -> Optional[float]:
        
        # Train the model
        with multiprocessing.Pool(processes=1) as pool:
            
            # Get the result from the process
            result = pool.apply_async(
                fit_model,
                args=(
                    model,
                    dataset.train.X,
                    dataset.train.y,
                    dataset.valid.X,
                    dataset.valid.y
                )
            )
            
            try:
                # Compute the accuracy on validation dataset
                reward = result.get(timeout=self._fitting_time_limit)

            # If the fitting time limit is exceeded,
            # set the reward to None
            except multiprocessing.TimeoutError:
                reward = None
        
        return reward
    
def fit_model(
        model: Model, 
        X_train: ndarray, 
        y_train: ndarray,
        X_valid: ndarray,
        y_valid: ndarray
    ) -> float:

    # Fit the model
    model.fit(X_train, y_train)
    
    # Compute the accuracy on validation dataset
    y_pred = model.predict(X_valid)
    reward = accuracy_score(y_valid, y_pred)
    
    return reward
