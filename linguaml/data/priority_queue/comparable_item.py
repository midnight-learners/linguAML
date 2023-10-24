from typing import Self, Callable, Any
from functools import total_ordering

@total_ordering
class ComparableItem:
    
    def __init__(self, item: Any) -> None:
        
        self._item = item
    
    def __eq__(self, other: Self) -> bool:
        
        try:
            return self.priority == other.priority
        except:
            return getattr(self._item, "__eq__")(other._item)
            

    def __lt__(self, other: Self) -> bool:
        
        try:
            return self.priority < other.priority
        except:
            return getattr(self._item, "__lt__")(other._item)
    
    def __str__(self) -> str:
        
        return str(self._item)
        
    @property
    def item(self) -> Any:
        """The wrapped item.
        """
        
        return self._item
    
    @property
    def priority(self) -> float:
        """The item's priority, which will be referred to during comparison.
        """
        
        return self.__class__.get_priority(self._item)
    
    @staticmethod
    def get_priority(item: Any) -> float:
        """A function that computes the priority of the given item.

        Parameters
        ----------
        item : Any
            Item.

        Returns
        -------
        Any
            Priority.
        """
        
        raise NotImplementedError
