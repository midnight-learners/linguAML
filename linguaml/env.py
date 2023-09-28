from typing import Optional, Iterable
from collections import deque
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from .data.utils import train_valid_test_split
from .data.dataset import Dataset
from .families import Family

class Env:
    
    def __init__(
            self,
            family: Family,
            hp_bounds: dict[str, tuple],
            dataset: Dataset,
            *,
            valid_size: float = 0.2,
            test_size: float = 0.2,
            state_dim: int = 10,
            random_state: Optional[int] = None,
        ) -> None:
        
        # Family of the models to fine tune
        self._family = family
        
        # Lower and upper bounds of all HPs
        self._hp_bounds = hp_bounds
        
        # Data
        X = dataset.features.to_numpy()
        y = dataset.targets.to_numpy().flatten()
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(y)
        
        # Random state
        self._random_state = random_state
        
        # Split into training, validation and test datasets
        split = train_valid_test_split(
            X, y,
            valid_size=valid_size,
            test_size=test_size,
            random_state=random_state
        )
        self._X_train, self._y_train = split["train"]
        self._X_valid, self._y_valid = split["valid"]
        self._X_test, self._y_test = split["test"]
        
        # State dimention
        self._state_dim = state_dim
        
        # A buffer of actions taken
        self._actions_taken = deque(maxlen=state_dim)
        
        # Reset env
        self._init_state = self.reset()
    
    @property
    def family(self) -> Family:
        """Family of the models to fine-tune.
        """
        
        return self._family
    
    @property
    def hp_bounds(self) -> dict[str, tuple]:
        """Lower and upper bounds of hyperparameters of the model family.
        """
        
        return self._hp_bounds
    
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
            sigmoid(rng.randn(self._family.n_hps)) 
            for _ in range(self._state_dim)
        ]
        
        # Reset actions taken
        self._actions_taken.clear()
        self._actions_taken.extend(random_actions)
        
        # Create an initial state
        init_state = np.array(self._actions_taken)
        self._init_state = init_state
        
        return init_state
        
    
    def step(self, action: Iterable[float]) -> tuple[np.ndarray, float]:
        
        # Generate the next state
        self._actions_taken.append(action)
        state = np.array(self._actions_taken)
        
        # Create the HP configuation from the action taken
        model = self._family.define_model(
            action=action,
            hp_bounds=self._hp_bounds
        )
        
        # Train the model
        model.fit(self._X_train, self._y_train)
        
        # Compute the accuracy on validation dataset
        y_pred = model.predict(self._X_valid)
        reward = accuracy_score(self._y_valid, y_pred)
        
        return state, reward
