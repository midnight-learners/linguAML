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
