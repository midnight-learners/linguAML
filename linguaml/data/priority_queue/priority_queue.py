from typing import Iterable
from heapq import heapify, heappush, heappop, nsmallest
from .ordered_item import OrderedItem

class PriorityQueue:
    
    def __init__(self) -> None:
        """Initializes an empty priority queue of ordered items.
        """
        
        # Internal data
        self._data = []

    def __len__(self) -> int:
        
        return len(self._data)
    
    def is_empty(self) -> bool:
        """Checks whether the queue is empty.

        Returns
        -------
        bool
            True if the queue is empty.
        """
        
        return len(self) == 0
        
    def push(self, item: OrderedItem) -> None:
        """Pushes an item into the queue.

        Parameters
        ----------
        item : OrderedItem
            Ordered item.
        """
        
        heappush(self._data, item)
        
    def pop(self) -> dict:
        """Pops the item with the highest priority.

        Returns
        -------
        dict
            The item with the highest priority.
        """
        
        item = heappop(self._data)
        
        return item
    
    def peek_first_n_items(self, n: int) -> list[dict]:
        """Gets the first several items with highest priorities.

        Parameters
        ----------
        n : int
            Maximum number of items to return.

        Returns
        -------
        list[dict]
            A list of dictionaries.
        """
        
        # Find the first several heap items with highest priorities
        items = nsmallest(n, self._data)
        
        return items
    
    def extend(self, iterable: Iterable[OrderedItem]) -> None:
        """Extends the queue with a collection of iterable ordered items.

        Parameters
        ----------
        iterable : Iterable[OrderedItem]
            A collection of iterable dictionaries with 
            priority values.
        """
        
        # Extend the list
        self._data.extend(iterable)
        
        # Make it a heap
        heapify(self._data)
