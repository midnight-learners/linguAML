from typing import Optional, Iterable
import multiprocessing
from collections import deque
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from .data.dataset import Dataset
from .families.base import Family
from .hp.bounds import NumericHPBounds
from .action import (
    calc_action_dim,
    convert_action_to_hp_config
)

class Env:
    
    def __init__(
            self,
            dataset: Dataset,
            family: Family,
            numeric_hp_bounds: NumericHPBounds | dict[str, tuple],
            *,
            state_dim: int = 10,
            random_state: Optional[int] = None,
        ) -> None:
        
        # Family of the models to fine tune
        self._family = family
        
        # Lower and upper bounds of all HPs
        if not isinstance(numeric_hp_bounds, NumericHPBounds):
            numeric_hp_bounds = NumericHPBounds.from_dict(numeric_hp_bounds)
        self._numeric_hp_bounds = numeric_hp_bounds
        
        # Dimensiton of agent's action space
        self._action_dim = calc_action_dim(family)

        # Random state
        self._random_state = random_state
        
        # Dataset
        self._dataset = dataset
        self._X_train, self._y_train = self._dataset.train.features.to_numpy(), self._dataset.train.targets.to_numpy().flatten()
        self._X_valid, self._y_valid = self._dataset.valid.features.to_numpy(), self._dataset.valid.targets.to_numpy().flatten()
        self._X_test, self._y_test = self._dataset.test.features.to_numpy(), self._dataset.test.targets.to_numpy().flatten()
        
        # Encode targets
        label_encoder = LabelEncoder()
        self._y_train = label_encoder.fit_transform(self._y_train)
        self._y_valid = label_encoder.transform(self._y_valid)
        self._y_test = label_encoder.transform(self._y_test)
        
        # State dimention
        self._state_dim = state_dim
        
        # A buffer of actions taken
        self._actions_taken = deque(maxlen=state_dim)
        
        # Reset env
        self._init_state = self.reset()
        
    @property
    def dataset(self) -> Dataset:
        """Dateset to work on.
        """
        
        return self._dataset
    
    @property
    def family(self) -> Family:
        """Family of the models to fine-tune.
        """
        
        return self._family
    
    @property
    def numeric_hp_bounds(self) -> NumericHPBounds:
        """Lower and upper bounds of the numeric hyperparameters 
        of the model family.
        """
        
        return self._numeric_hp_bounds
    
    @property
    def state_dim(self) -> int:
        """State dimension.
        """
        
        return self._state_dim
    
    @property
    def init_state(self) -> np.ndarray:
        """Initial state.
        """
        
        return self._init_state
    
    def reset(self) -> np.ndarray:
        
        # Sigmoid function
        sigmoid = lambda x: 1 / (1 + np.exp(-x))
        
        # NumPy's random generator
        rng = np.random.RandomState(seed=self._random_state)
        
        # Generate random actions
        random_actions = [
            sigmoid(rng.randn(self._action_dim)) 
            for _ in range(self._state_dim)
        ]
        
        # Reset actions taken
        self._actions_taken.clear()
        self._actions_taken.extend(random_actions)
        
        # Create an initial state
        init_state = np.array(self._actions_taken)
        self._init_state = init_state
        
        return init_state
    
    def step(self, action: Iterable[float]) -> tuple[np.ndarray, Optional[float]]:
        
        # Generate the next state
        self._actions_taken.append(action)
        state = np.array(self._actions_taken)
        
        # Create the hyperparameter configuation from the action taken
        hp_config = convert_action_to_hp_config(
            action=action,
            family=self._family,
            numeric_hp_bounds=self._numeric_hp_bounds
        )
        
        # Difine the model with the hyperparameter configuation
        model = self._family.define_model(
            hp_config=hp_config
        )
        
        # Train the model
        with multiprocessing.Pool(processes=1) as pool:
            
            result = pool.apply_async(
                fit_model,
                args=(
                    model,
                    self._X_train,
                    self._y_train,
                    self._X_valid,
                    self._y_valid
                )
            )
            
            try:
                # Compute the accuracy on validation dataset
                reward = result.get(timeout=5.0)
                
            except multiprocessing.TimeoutError:
                reward = None
        
        return state, reward
    
def fit_model(model, X_train, y_train, X_valid, y_valid):
            
    model.fit(X_train, y_train)
    
    # Compute the accuracy on validation dataset
    y_pred = model.predict(X_valid)
    reward = accuracy_score(y_valid, y_pred)
    
    return reward
