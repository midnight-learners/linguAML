from pydantic import Field
from sklearn.ensemble import RandomForestClassifier

from linguaml.hp import HPConfig, CategoricalHP
from .base import define_family_type

class Criterion(CategoricalHP):
    
    GINI = "gini"
    EMTROPY = "entropy"
    LOG_LOSS = "log_loss"

class RandomForestClassifierConfig(HPConfig):
    
    n_estimators: int = Field(
        description="""
        The number of trees in the forest.
        [A positive integer]
        """
    )
    
    criterion: Criterion = Field(
        f"""
        The function to measure the quality of a split. Supported criteria are “gini” for the Gini impurity and “log_loss” and “entropy” both for the Shannon information gain, see Mathematical formulation. Note: This parameter is tree-specific.
        [Possible values: {Criterion.level_values()}]
        """
    )
    
    max_depth: int
    
    min_samples_split: int = Field(
        """
        The minimum number of samples required to split an internal node:
        [A postive integer]
        """
    )

RandomForestClassifierFamily = define_family_type(
    name="RandomForestClassifierFamily",
    model_type=RandomForestClassifier,
    hp_config_type=RandomForestClassifierConfig
)
