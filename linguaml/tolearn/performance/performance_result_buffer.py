from typing import Optional

# Imports from this package
from linguaml.collections import PriorityQueue
from .performance_result import PerformanceResult

class PerformanceResultBuffer:
    
    def __init__(
            self, 
            capacity: int,
            internal_capacity: Optional[int] = None
        ) -> None:
        """Initialize a performance result buffer.

        Parameters
        ----------
        capacity : int
            The capacity of the buffer.
        internal_capacity : Optional[int], optional
            The internal capacity of the buffer, by default None.
            If None, then the internal capacity is set to 2 times the capacity.
        """
        
        self._capacity = capacity
        
        if internal_capacity is None:
            internal_capacity = capacity * 2
        self._internal_capacity = internal_capacity
        
        # High performance results stored in a priority queue
        self._results = PriorityQueue(
            get_priority=lambda result: -result.score
        )
        
    @property
    def capacity(self) -> int:
        """The capacity of the buffer.
        """
        
        return self._capacity
    
    def push(self, result: PerformanceResult) -> None:
        """Push a new result to the buffer.
        
        Parameters
        ----------
        result : PerformanceResult
            The result to be pushed.
        """
        
        self._results.push(result)
        
        # If the queue is full (the internal capacity is reached), 
        # remove the low performance results
        if len(self._results) > self._internal_capacity:
            self._reorganize()
            
    def extend(self, results: list[PerformanceResult]) -> None:
        """Extend the buffer with a list of results.
        
        Parameters
        ----------
        results : list[PerformanceResult]
            The results to be pushed.
        """
        
        self._results.extend(results)
        
        # If the queue is full (the internal capacity is reached), 
        # remove the low performance results
        if len(self._results) > self._internal_capacity:
            self._reorganize()
            
    def peek_first_n_high_performance_results(self, n: int) -> list[PerformanceResult]:
        """Peek the first n high performance results in the buffer.
        
        Parameters
        ----------
        n : int
            The number of results to be peeked.

        Returns
        -------
        list[PerformanceResult]
            The first n high performance results in the buffer.
            If n is greater than the capacity, then return the results in the buffer.
        """
        
        # If n is greater than the capacity,
        # then return the results in the buffer
        if n > self._capacity:
            n = self._capacity
            
        return self._results.peek_first_n_items(n)
        
    def to_list(self) -> list[PerformanceResult]:
        """Return the results in the buffer as a list.

        Returns
        -------
        list[PerformanceResult]
            The results in the buffer ordered by score in descending order.
        """
        
        # Note that the number of results is limited by the capacity
        n_results = min(len(self._results), self._capacity)
        
        return list(self._results.peek_first_n_items(n_results))
    
    def _reorganize(self) -> None:
        """Reorganize the buffer by throwing away the low performance results.
        """
        
        # Throw away the low performance results
        results_to_keep = self._results.peek_first_n_items(self._capacity)
        
        # Clear the queue
        self._results.clear()
        
        # Push the high performance results back to t
        # he queue
        self._results.extend(results_to_keep)
