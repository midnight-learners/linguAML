from collections import deque
from torch.utils.data import Dataset

from ..env import Env

class ReplayBuffer(deque, Dataset):
    
    def __init__(
            self,
            env: Env,
            capacity: int
        ) -> None:
        
        super().__init__(maxlen=capacity)

        self._env = env
        self._capacity = capacity
    
    @property
    def capacity(self) -> int:
        return self._capacity
