from .base import Family
from .svc import SVCFamily

def get_family(name: str) -> Family:
    """Gets the model family bi its name.

    Parameters
    ----------
    name : str
        Model family name.

    Returns
    -------
    Family
        _description_

    Raises
    ------
    ValueError
        _description_
    """
    
    # Convert to lower case
    name = name.lower().strip()
    
    match name:
        
        case "svc":
            return SVCFamily
        
        case _:
            raise ValueError(f"unkown model family: {name}")
        