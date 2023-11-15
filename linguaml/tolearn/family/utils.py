from .family import Family

def get_family(name: str) -> Family:
    """Gets the model family by its name.

    Parameters
    ----------
    name : str
        Model family name.

    Returns
    -------
    Family
        An abstract class of the model family.
    """
    
    # Convert to lower case
    name = name.lower().strip()
    