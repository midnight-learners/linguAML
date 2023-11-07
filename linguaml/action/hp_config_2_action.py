import numpy as np

# Imports from this package
from linguaml.hp import HPConfig, CategoricalHP
from linguaml.hp.bounds import NumericHPBounds

def convert_hp_config_to_action(
        hp_config: HPConfig,
        numeric_hp_bounds: NumericHPBounds
    ) -> np.ndarray:
    """Converts a hyperparameter configuration to an action.

    Parameters
    ----------
    hp_config : HPConfig
        Hyperparameter configuration.
        
    numeric_hp_bounds : NumericHPBounds
        Numeric hyperparameter bounds.

    Returns
    -------
    np.ndarray
        A single action represented as a NumPy array.
    """
    
    # Get continuous action
    cont_action = []
    for hp_name in hp_config.numeric_hp_names():
        hp_value = getattr(hp_config, hp_name)
        hp_max = numeric_hp_bounds.name2bounds[hp_name].max
        hp_min = numeric_hp_bounds.name2bounds[hp_name].min
        action_value = (hp_value - hp_min) / (hp_max - hp_min)
        cont_action.append(action_value)
    cont_action = np.array(cont_action)
    
    # Get discrete actions
    disc_actions = []
    for hp_name in hp_config.categorical_hp_names():
        hp_value: CategoricalHP = getattr(hp_config, hp_name)
        disc_actions.append(hp_value.one_hot)
    disc_actions = np.concatenate(disc_actions)
    
    # Generate the final action
    action = np.concatenate([cont_action, disc_actions])
    
    return action
