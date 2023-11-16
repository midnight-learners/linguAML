from typing import Optional, Generator
from sklearn import utils

# Imports from this package
from .replay_buffer import ReplayBuffer
from ..transition import Transition, BatchedTransitions

class ReplayBufferLoader:
    
    def __init__(
            self, 
            replay_buffer: ReplayBuffer,
            batch_size: int,
            min_batch_size: Optional[int] = None,
            shuffle: bool = True,
            random_state: Optional[int] = None
        ) -> None:
        
        # Convert to list
        replay_buffer = list(replay_buffer)
        
        # Shuffle the replay buffer if required
        if shuffle:
            replay_buffer = utils.shuffle(replay_buffer, random_state=random_state)

        self._replay_buffer = replay_buffer
        self._batch_size = batch_size
        
        if min_batch_size is None:
            min_batch_size = batch_size
        self._min_batch_size = min_batch_size
        
    def gen_batches(self) -> Generator[BatchedTransitions, None, None]:
        """Generate batches of transitions.
        In fact, it yields a `BatchedTransitions` instance each time.

        Yields
        ------
        Generator[BatchedTransitions, None, None]
            A generator of `BatchedTransitions` instances.
        """
        
        for slice in utils.gen_batches(
                len(self._replay_buffer), 
                batch_size=self._batch_size, 
                min_batch_size=self._min_batch_size
            ):
            transitions: list[Transition] = self._replay_buffer[slice.start:slice.stop:slice.step]
            batched_transitions = BatchedTransitions.from_transitions(transitions)
            yield batched_transitions
