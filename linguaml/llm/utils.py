import json

# Imports from this package
from linguaml.tolearn.family import Family

def get_hp_config_description(family: Family) -> str:
    """Get the description of the hyperparameter configuration.
    
    Parameters
    ----------
    family : Family
        The model family.
        
    Returns
    -------
    str
        The description of the hyperparameter configuration.
    """
    
    
    lines = []
    
    hp_name_to_description = family.hp().description()
    for hp_name in hp_name_to_description:
        description = hp_name_to_description[hp_name]
        lines.append(f"{hp_name}: {description}")
    
    description = "\n".join(lines)
    
    return description

def get_hp_output_format(family: Family) -> str:
    """Get the output format of the hyperparameter configuration.
    
    Parameters
    ----------
    family : Family
        The model family.
        
    Returns
    -------
    str
        The output format of the hyperparameter configuration.
        For example, '{"C": "...", "kernel": "...", "gamma": "...", "tol": "..."}'.
    """
    
    output_template = {hp_name: "..." for hp_name in family.hp_names()}
    output_format = json.dumps(output_template)
    
    return output_format
