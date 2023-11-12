from .types import Number

def is_number_list(object: object) -> bool:
    """Check if an object is a vector.

    Parameters
    ----------
    object : object
        An object.

    Returns
    -------
    bool
        True if the object is a vector, False otherwise.
    """
    
    if not isinstance(object, list):
        return False
    
    return all(map(
        lambda x: isinstance(x, Number), 
        object
    ))
