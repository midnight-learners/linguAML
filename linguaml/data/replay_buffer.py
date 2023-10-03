from collections import deque
from torch.utils.data import Dataset

class ReplayBuffer(deque, Dataset):
    
    def __init__(
            self,
            capacity: int
        ) -> None:
        
        super().__init__(maxlen=capacity)

        self._capacity = capacity
    
    @property
    def capacity(self) -> int:
        return self._capacity
