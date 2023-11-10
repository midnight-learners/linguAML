from pydantic import Field
from lightgbm import LGBMClassifier
from linguaml.hp import HPConfig, CategoricalHP
from .base import define_family_type

class BoostingType(CategoricalHP):

    GBDT = 'gbdt'
    DART = 'dart'
    RF = 'rf'

class LGBMClassifierConfig(HPConfig):
    
    boosting_type: BoostingType = Field(
        description=f"""
        ‘gbdt’, traditional Gradient Boosting Decision Tree. ‘dart’, Dropouts meet Multiple Additive Regression Trees. ‘rf’, Random Forest.
        [Possible values: {BoostingType.level_values()}]
        """
    )

    num_leaves: int = Field(
        description="""
        Maximum tree leaves for base learners.
        [A positive integer]
        """
    )

    max_depth: int = Field(                 ########## non-negative acceptable?
        description="""
        Maximum tree depth for base learners, <=0 means no limit.
        [A non-negative integer]
        """
    )
    
    n_estimators: int = Field(
        description="""
        Number of boosted trees to fit.
        [A positive integer]
        """
    )

    min_split_gain: float = Field(                ########## non-negative acceptable?
        description="""
        Minimum loss reduction required to make a further partition on a leaf node of the tree.
        [Non-negative number] 
        """
    )

    min_child_weight: float = Field(                ########## non-negative acceptable?
        description="""
        Minimum sum of instance weight (Hessian) needed in a child (leaf).
        [Non-negative number]
        """
    )

    min_child_samples: int = Field(
        description="""
        Minimum number of data needed in a child (leaf).
        [A positive integer]
        """
    )

    reg_alpha: float = Field(
        description="""
        L1 regularization term on weights.
        [Positive number]
        """
    )

    reg_lambda: float = Field(                  ########## not sure if this can be set together with reg_alpha
        description="""
        L2 regularization term on weights.
        [Positive number]
        """
    )

LGBMClassifierFamily = define_family_type(
    name="LGBMClassifierFamily",
    model_type=LGBMClassifier,
    hp_config_type=LGBMClassifierConfig
)


"""
Other possible parameters:
- learning_rate (float, optional (default=0.1)) – Boosting learning rate. You can use callbacks parameter of fit method to shrink/adapt learning rate in training using reset_parameter callback. Note, that this will ignore the learning_rate argument in training.
- subsample_for_bin (int, optional (default=200000)) – Number of samples for constructing bins.
- subsample (float, optional (default=1.)) – Subsample ratio of the training instance.
- subsample_freq (int, optional (default=0)) – Frequency of subsample, <=0 means no enable.
- colsample_bytree (float, optional (default=1.)) – Subsample ratio of columns when constructing each tree.
"""