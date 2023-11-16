from pydantic import Field
from sklearn.ensemble import RandomForestClassifier

# Imports from this package
from ..hp import HPConfig, CategoricalHP
from .model_family import ModelFamily

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
        description=f"""
        The function to measure the quality of a split. Supported criteria are “gini” for the Gini impurity and “log_loss” and “entropy” both for the Shannon information gain, see Mathematical formulation. Note: This parameter is tree-specific.
        [Possible values: {Criterion.level_values()}]
        """
    )
    
    max_depth: int = Field(
        description=f"""
        The maximum depth of the tree.
        [A positive integer]
        """
    )
    
    min_samples_split: int = Field(
        description="""
        The minimum number of samples required to split an internal node:
        [A postive integer]
        """
    )
    
    min_samples_leaf: int = Field(
        description="""
        The minimum number of samples required to be at a leaf node. A split point at any depth will only be considered if it leaves at least min_samples_leaf training samples in each of the left and right branches. This may have the effect of smoothing the model, especially in regression.
        [A positive integer]
        """
    )
    
    max_features: float = Field(
        description="""
        The fraction of features to consider when looking for the best split.
        [A real number in between 0 and 1]
        """
    )
        

random_forest_classifier_family = ModelFamily(
    hp_config_type=RandomForestClassifierConfig,
    model_type=RandomForestClassifier,
)
