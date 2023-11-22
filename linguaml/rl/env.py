from typing import Optional
from collections import deque
import multiprocessing
from numpy import ndarray
import numpy as np
from sklearn.metrics import accuracy_score, f1_score

# Imports from this package
from linguaml.types import Model
from linguaml.data.dataset import Dataset
from linguaml.tolearn.performance import PerformanceMetric
from .action import Action
from .state import StateConfig, State

class Env:
    
    def __init__(
            self,
            datasets: list[Dataset],
            *,
            performance_metric: PerformanceMetric = PerformanceMetric.ACCURACY,
            lookback: int = 10,
            fitting_time_limit: float = 5.0,
            random_state: Optional[int] = None,
        ) -> None:
        
        # Performance metric
        self._performance_metric = performance_metric
        
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
        
        # Initial state
        self._initial_state: Optional[State] = None
        
    @property
    def datasests(self) -> list[Dataset]:
        """Datesets to work on.
        """
        
        return self._datasets
    
    @property
    def initial_state(self) -> Optional[State]:
        """Initial state.
        None if the environment has never been reset.
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
                    dataset.valid.y,
                    self._performance_metric
                )
            )
            
            try:
                # Compute the score on validation dataset
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
        y_valid: ndarray,
        performance_metric: PerformanceMetric = PerformanceMetric.ACCURACY,
    ) -> float:

    # Fit the model
    model.fit(X_train, y_train)
    
    # Compute the accuracy on validation dataset
    y_pred = model.predict(X_valid)
    
    # Compute the reward based on the performance metric
    match performance_metric:
        
        case PerformanceMetric.ACCURACY:
            reward = accuracy_score(y_valid, y_pred)
        
        case PerformanceMetric.F1_SCORE:
            reward = f1_score(y_valid, y_pred, average="macro")

    return reward
