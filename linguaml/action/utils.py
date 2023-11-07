from linguaml.families.base import Family

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
