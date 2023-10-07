import numpy as np
from torch import Tensor
from .hp import HPConfig
from .families.base import Family

def calc_action_dim(family: Family) -> int:
    """Calculates the dimension of the action space.

    Parameters
    ----------
    family : Family
        Model family.

    Returns
    -------
    int
        Dimension of the action space.
    """
    
    action_dim = family.n_numeric_hps()
    for name in family.categorical_hp_names():
        n_levels = family.n_levels_in_category(name)
        action_dim += n_levels
        
    return action_dim

def extract_cont_action(
        action: Tensor | np.ndarray,
        family: Family
    ) -> Tensor | np.ndarray:
    """Extracts the continuous action from the overall action.

    Parameters
    ----------
    action : Tensor | np.ndarray
        Overall action with continuous and discrete actions concatenated.
    family : Family
        Model family.

    Returns
    -------
    Tensor | np.ndarray
        A tensor or NumPy array of real numbers in between 0 and 1 representing 
        the continuous action.
    """
    
    # If the action is not a tensor nor a NumPy array,
    # then we try to convert it to a NumPy array
    if not isinstance(action, (Tensor, np.ndarray)):
        action = np.array(action, dtype=np.float32)
    
    return action[..., :family.n_numeric_hps()]

def extract_disc_action(
        action: Tensor | np.ndarray,
        family: Family,
        categorical_hp_name: str
    ) -> Tensor | np.ndarray:
    """Extracts the discrete action from the overall action.

    Parameters
    ----------
    action : Tensor | np.ndarray
        Overall action with continuous and discrete actions concatenated.
    family : Family
        Model family.
    categorical_hp_name : str
        Name of the categorical hyperparameter.

    Returns
    -------
    Tensor | np.ndarray
        A tensor or NumPy array of integers representing 
        the indices of levels of all categories.
    """
    
    # Handle tensor
    if isinstance(action, Tensor):
        return _extract_disc_action_tensor(action, family, categorical_hp_name)
    
    # Handle NumPy array
    elif isinstance(action, np.ndarray):
        return _extract_disc_action_numpy(action, family, categorical_hp_name)
    
    # If the action is not a tensor nor a NumPy array,
    # then we try to convert it to a NumPy array
    if not isinstance(action, (Tensor, np.ndarray)):
        action = np.array(action, dtype=np.float32)
        
    return _extract_disc_action_numpy(action, family, categorical_hp_name)

def convert_action_to_hp_config(
        action: Tensor | np.ndarray,
        family: Family,
        cont_hp_bounds: dict[str, tuple]
    ) -> HPConfig:
    """Converts the (single) input action to 
    a hyperparameter configuration.
    
    Notes
    -----
    The batched actions input is not supported.

    Parameters
    ----------
    action : Tensor | np.ndarray
        Single overall action.
    family : Family
        Model family.
    cont_hp_bounds : dict[str, tuple]
        Lower and upper bounds of all continuous hyparameters.

    Returns
    -------
    HPConfig
        Hyperparameter configuration.
    """
    
    assert len(action.shape) == 1,\
        "only single action input is supported"
    
    hps = {}
    
    # Extract the continuous action
    cont_action = extract_cont_action(action, family)
    
    # Iterate over all slots in the action
    for i, hp in enumerate(cont_action):
        
        # Find all numberic hyperparameters
        hp_name = family.numeric_hp_names()[i]
        
        # Get the HP type, which is either float or int
        hp_type = family.hp_type(hp_name)
        
        # Lower and upper bounds
        hp_min, hp_max = cont_hp_bounds[hp_name] 
        
        # Recover the hyperparameter value        
        hp = hp_type(hp * (hp_max - hp_min) + hp_min)
        
        hps[hp_name] = hp
        
    # Extract the discrete actions
    for hp_name in family.categorical_hp_names():
        
        # The index of the level of the category
        disc_action: int = extract_disc_action(action, family, hp_name)
        
        # Categorical hyperparameter value
        categorical_hp_type = family.hp_type(hp_name)
        hp = categorical_hp_type.from_index(disc_action)
        
        hps[hp_name] = hp
    
    # Create the instances by unpacking `hps`
    hp_config = family.hp()(**hps)
    
    return hp_config

def _extract_disc_action_tensor(
        action: Tensor,
        family: Family,
        categorical_hp_name: str
    ) -> Tensor:
    """Extracts the discrete action from the input tensor.

    Parameters
    ----------
    action : Tensor
        Overall action with continuous and discrete actions concatenated.
    family : Family
        Model family.
    categorical_hp_name : str
        Name of the categorical hyperparameter.

    Returns
    -------
    Tensor
        A tensor of integers representing 
        the indices of levels of all categories.
    """
    
    assert categorical_hp_name in family.categorical_hp_names(),\
        f"unknown category '{categorical_hp_name}'"
    
    start = family.n_numeric_hps()
    
    for name in family.categorical_hp_names():
        
        if name == categorical_hp_name:
            break
        
        start += family.n_levels_in_category(name)
    
    end = start + family.n_levels_in_category(categorical_hp_name)
    
    # One-hot encoding vector representing 
    # the choice of the category level
    one_hot_action = action[..., start:end]
    
    # The level index
    action = one_hot_action.max(dim=-1).indices
    
    return action

def _extract_disc_action_numpy(
        action: np.ndarray,
        family: Family,
        categorical_hp_name: str
    ) -> np.ndarray:
    """Extracts the discrete action from the input NumPy array.

    Parameters
    ----------
    action : np.ndarray
        Overall action with continuous and discrete actions concatenated.
    family : Family
        Model family.
    categorical_hp_name : str
        Name of the categorical hyperparameter.

    Returns
    -------
    np.ndarray
        A NumPy array of integers representing 
        the indices of levels of all categories.
    """
    
    assert categorical_hp_name in family.categorical_hp_names(),\
        f"unknown category '{categorical_hp_name}'"
    
    start = family.n_numeric_hps()
    
    for name in family.categorical_hp_names():
        
        if name == categorical_hp_name:
            break
        
        start += family.n_levels_in_category(name)
    
    end = start + family.n_levels_in_category(categorical_hp_name)
    
    # One-hot encoding vector representing 
    # the choice of the category level
    one_hot_action = action[..., start:end]
    
    # The level index
    action = one_hot_action.argmax(axis=-1)
    
    return action
