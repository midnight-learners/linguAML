import numpy as np
from scipy import stats

# Imports from this package
from linguaml.hp import HPConfig
from linguaml.hp.bounds import NumericHPBounds
from linguaml.families.base import Family
from .action.action_2_hp_config import convert_action_to_hp_config

def generate_random_action(
        family: Family,
        cont_action_noise_scale: float = 0.5,
    ) -> np.ndarray:
    """Generate a random action.
    
    Parameters
    ----------
    family : Family
        Model family.
        
    Returns
    -------
    np.ndarray
        A random action.
    """
    
    # Generate a continuous action
    # associated with numeric hyperparameters
    n = family.n_numeric_hps()
    cont_action = stats.norm(loc=0.5, scale=cont_action_noise_scale).rvs(size=n)
    cont_action = np.clip(cont_action, 0.0, 1.0)
    
    # Generate discrete actions 
    # associsated with categorical hyperparameters
    disc_actions = []
    for categorical_hp_name in family.categorical_hp_names():
        n_levels = family.n_levels_in_category(categorical_hp_name)
        I = np.eye(n_levels)
        disc_action = I[stats.randint.rvs(low=0, high=n_levels, size=1).item()]
        disc_actions.append(disc_action)
    disc_action = np.concatenate(disc_actions)
    
    # Concatenate the continuous and discrete actions
    action = np.concatenate([cont_action, disc_action])
    
    return action

def generate_random_hp_config(
        family: Family,
        numeric_hp_bounds: NumericHPBounds
    ) -> HPConfig:
    """Generate a random hyperparameter configuration.
    
    Parameters
    ----------
    family : Family
        Model family.
    
    numeric_hp_bounds : NumericHPBounds
        Lower and upper bounds of the numeric hyperparameters.

    Returns
    -------
    HPConfig
        A random hyperparameter configuration.
    """
    
    action = generate_random_action(family)
    hp_config = convert_action_to_hp_config(action, family, numeric_hp_bounds)
    
    return hp_config
    
    

def generate_random_hp_configs(
        n: int,
        family: Family,
        numeric_hp_bounds: NumericHPBounds
    ) -> list[HPConfig]:
    """Generate a list of random hyperparameter configurations.
        
    Parameters
    ----------
    n : int
        The number of random hyperparameter configurations to generate.

    family : Family
        Model family.
    
    numeric_hp_bounds : NumericHPBounds
        Lower and upper bounds of the numeric hyperparameters.

    Returns
    -------
    list[HPConfig]
        A list of random hyperparameter configurations.
    """
    
    hp_configs = []
    for _ in range(n):
        hp_config = generate_random_hp_config(family, numeric_hp_bounds)
        hp_configs.append(hp_config)
        
    return hp_configs
