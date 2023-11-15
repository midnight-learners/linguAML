from numpy import ndarray 
import numpy as np

# Imports from this package
from linguaml.tolearn.family import Family
from linguaml.tolearn.hp import HPConfig, CategoricalHP
from linguaml.tolearn.hp.bounds import NumericHPBounds
from linguaml.tolearn.performance import PerformanceResult

def calc_n_state_features(family: Family) -> int:
    
    # Number of numeric hyperparameters
    n_numeric_hps = family.n_numeric_hps()
    
    # Sum of number of levels in all categorical hyperparameters
    n_levels = sum([
        family.n_levels_in_category(hp_name)
        for hp_name in family.categorical_hp_names()
    ])
    
    # 1 is for the reward
    n_state_features = n_numeric_hps + n_levels + 1
    
    return n_state_features

def encode_hp_config_and_accuracy(
        hp_config: HPConfig, 
        accuracy: float,
        numeric_hp_bounds: NumericHPBounds
    ) -> ndarray:
    """Encode a hyperparameter configuration and its accuracy into a row of the state matrix.

    Parameters
    ----------
    hp_config : HPConfig
        Hyperparameter configuration.
    accuracy : float
        Accuracy of the hyperparameter configuration on the validation set.
    numeric_hp_bounds : NumericHPBounds
        The bounds of the numerical hyperparameters.

    Returns
    -------
    ndarray
        A row of the state matrix.
    """
    
    # Continuous part
    # Normalization of numerical hyperparameters
    # Each numerical hyperparameter is normalized to [0, 1]
    entries = []
    for hp_name in hp_config.numeric_hp_names():
        hp = getattr(hp_config, hp_name)
        bounds = numeric_hp_bounds.get_bounds(hp_name)
        entry = (hp - bounds.min) / (bounds.max - bounds.min)
        entries.append(entry)
    continuous_part = np.array(entries)
    
    # Discrete part
    # One-hot encoding vectors of categorical hyperparameters
    one_hot_arrays = []
    for hp_name in hp_config.categorical_hp_names():
        hp_value: CategoricalHP = getattr(hp_config, hp_name)
        one_hot_arrays.append(hp_value.one_hot)
    discrete_part = np.concatenate(one_hot_arrays)
    
    # Reward
    # It is a singleton array containing the accuracy
    reward_singleton_array = np.array([accuracy])
    
    # Concatenate continuous part, discrete part, and accuracy
    encoded_array = np.concatenate((continuous_part, discrete_part, reward_singleton_array))
    
    return encoded_array

def encode_performance_result(
        performance_result: PerformanceResult,
        numeric_hp_bounds: NumericHPBounds
    ) -> ndarray:
    """Encode a performance result into a row of the state matrix.

    Parameters
    ----------
    performance_result : PerformanceResult
        Performance result.
    numeric_hp_bounds : NumericHPBounds
        The bounds of the numerical hyperparameters.

    Returns
    -------
    ndarray
        A row of the state matrix.
    """
    
    hp_config = performance_result.hp_config
    accuracy = performance_result.accuracy
    
    return encode_hp_config_and_accuracy(hp_config, accuracy, numeric_hp_bounds)

def decode_state_row(
        encoded_array: ndarray, 
        family: Family, 
        numeric_hp_bounds: NumericHPBounds
    ) -> tuple[HPConfig, float]:
    """Decodes a row of the state matrix.

    Parameters
    ----------
    encoded_array : ndarray
        A row of the state matrix.
    family : Family
        The family of models.
    numeric_hp_bounds : NumericHPBounds
        The bounds of the numerical hyperparameters.

    Returns
    -------
    tuple[HPConfig, float]
        The hyperparameter configuration and the accuracy.
    """
    
    # Hyperparameters
    hps = {}
    
    # Pointer to the current index in the encoded array
    start = 0
    
    # Decode continuous part
    for hp_name in family.hp().numeric_hp_names():
        bounds = numeric_hp_bounds.get_bounds(hp_name)
        value = encoded_array[start]
        
        # Recover the original value
        hp = value * (bounds.max - bounds.min) + bounds.min
        hps[hp_name] = hp
        
        # Update the pointer
        start += 1
    
    # Decode discrete part
    for hp_name in family.hp().categorical_hp_names():
        n_levels = family.hp().n_levels_in_category(hp_name)
        one_hot = encoded_array[start : start + n_levels]
        hp_type: CategoricalHP = family.hp().hp_type(hp_name)
        
        # Recover the original value
        hp = hp_type.from_one_hot(one_hot)
        hps[hp_name] = hp
        
        # Update the pointer
        start += n_levels
        
    # Recover the hyperparameter configuration
    hp_config = family.hp()(**hps)
    
    # Decode reward
    accuracy = encoded_array[start]
    
    return hp_config, accuracy
  