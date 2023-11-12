from typing import Optional, Iterable, Callable, Any
from heapq import heapify, heappush, heappop, nsmallest
from .comparable_item import ComparableItem

class PriorityQueue:
    
    def __init__(
            self,
            get_priority: Optional[Callable[[Any], float]] = None,
        ) -> None:
        """Initialize an empty priority queue of ordered items.
        """
        
        methods = {}
        if get_priority is not None:
            methods["get_priority"] = get_priority
        
        self._ComparableItem = type(
                "_ComparableItem",
                (ComparableItem,),
                methods
            )
        
        # Internal data
        self._data: list[self._ComparableItem] = []

    def __len__(self) -> int:
        
        return len(self._data)
    
    def is_empty(self) -> bool:
        """Check whether the queue is empty.

        Returns
        -------
        bool
            True if the queue is empty.
        """
        
        return len(self) == 0
        
    def push(self, item: Any) -> None:
        """Push an item into the queue.

        Parameters
        ----------
        item : Any
           An item.
        """
        
        heappush(self._data, self._ComparableItem(item))
        
    def pop(self) -> Any:
        """Pop the item with the highest priority.

        Returns
        -------
        Any
            The item with the highest priority.
        """
        
        comparable_item = heappop(self._data)
        
        item = comparable_item.item
        
        return item
    
    def peek_first_n_items(self, n: int) -> list[Any]:
        """Get the first several items with highest priorities.

        Parameters
        ----------
        n : int
            Maximum number of items to return.

        Returns
        -------
        list[Any]
            A list of items.
        """
        
        # Find the first several heap items with highest priorities
        comparable_items = nsmallest(n, self._data)
        
        # Convert the items
        items = list(map(
            lambda comparable_item: comparable_item.item,
            comparable_items
        ))
        
        return items
    
    def extend(self, iterable: Iterable[Any]) -> None:
        """Extend the queue with a collection of iterable ordered items.

        Parameters
        ----------
        iterable : Iterable[ComparableItem]
            A collection of iterable dictionaries with 
            priority values.
        """
        
        # Extend the list
        comparable_items = list(map(
            self._ComparableItem,
            iterable
        ))
        self._data.extend(comparable_items)
        
        # Make it a heap
        heapify(self._data)
        
    def clear(self) -> None:
        """Clear the queue.
        """
        
        self._data.clear()
