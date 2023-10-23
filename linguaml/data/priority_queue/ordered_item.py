from typing import Self, Any
from functools import total_ordering

@total_ordering
class OrderedItem:
    
    def __init__(self, item: Any) -> None:
        
        self._item = item
    
    def __eq__(self, other: Self) -> bool:
        
        return self.value == other.value
        
    
    def __lt__(self, other: Self) -> bool:
        
        return self.value < other.value
    
    def __str__(self) -> str:
        
        return str(self._item)
    
    def __repr__(self) -> str:
        
        return (
            "(item={item}, "
            "value={value})"
        ).format(
            item=self._item,
            value=self.value
        )
    
    @property
    def value(self) -> float:
        """The item's value, which will be referred to during comparison.
        """
        
        return self.__class__.get_value(self._item)
    
    @staticmethod
    def get_value(item: Any) -> Any:
        """A function that computes the value of the given item.

        Parameters
        ----------
        item : Any
            Item.

        Returns
        -------
        Any
            A comparable value.
        """
        
        return item
