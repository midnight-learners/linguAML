from collections import deque
from linguaml.rl.transition import Transition

class ReplayBuffer(deque):
    
    def __init__(
            self,
            capacity: int
        ) -> None:
        
        super().__init__(maxlen=capacity)

        self._capacity = capacity
    
    @property
    def capacity(self) -> int:
        """Maximum number of transitions the buffer can hold.
        """
        
        return self._capacity
    
    def add(self, transition: Transition) -> None:
        """Add a transition to the buffer.

        Parameters
        ----------
        transition : Transition
            A transition.
        """
        
        # Add the new transition to the buffer
        self.append(transition)
    
    def extend(self, transitions: list[Transition]) -> None:
        """Add a list of transitions to the buffer.

        Parameters
        ----------
        transitions : list[Transition]
            A list of transitions.
        """
        
        # Add the new transitions to the buffer
        super().extend(transitions)
        
