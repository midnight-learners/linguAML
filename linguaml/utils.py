import re
from pathlib import Path
import inflection
from torch import Tensor
from .families.base import Family


def mkdir_if_not_exists(dir: Path) -> Path:
    
    # Make dir if it does not exist
    if not dir.is_dir():
        dir.mkdir(parents=True, exist_ok=True)
    
    return dir

def dasherize(text: str) -> str:
    
    # Convert to lower case
    text = text.lower()
    
    # Replace special characters with whitespaces
    text = re.sub(r"[^a-z0-9]", " ", text).strip()
    
    # Replace all whitespaces with dashes
    text = re.sub(r"\s+", "-", text)
    
    # Finalize
    text = inflection.dasherize(text)
    
    return text

def extract_cont_action(
        action: Tensor,
        family: Family
    ) -> Tensor:
    
    return action[..., :family.n_numeric_hps()]

def extract_disc_action(
        action: Tensor,
        family: Family,
        categorical_hp_name: str
    ) -> Tensor:
    
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
    
def calc_action_dim(family: Family) -> int:
    
    action_dim = family.n_numeric_hps()
    for name in family.categorical_hp_names():
        n_levels = family.n_levels_in_category(name)
        action_dim += n_levels
        
    return action_dim
