from pydantic import Field
from xgboost import XGBClassifier
from linguaml.hp import HPConfig, CategoricalHP
from .base import define_family_type

class Booster(CategoricalHP):

    GBTREE = 'gbtree'
    GBLINEAR = 'gblinear'
    DART = 'dart'

class XGBClassifierConfig(HPConfig):

    n_estimators: int = Field(
        description="""
        Number of boosting rounds.
        [A positive integer]
        """
    )

    max_leaves: int = Field(                ########## non-negative acceptable?
        description="""
        Maximum number of leaves; 0 indicates no limit.
        [A non-negative integer]
        """
    )

    max_depth: int = Field(                 ########## set to be positive here. this is optional.
        description="""
        Maximum tree depth for base learners.
        [A positive integer]
        """
    )
    
    booster: Booster = Field(
        description=f"""
        Specify which booster to use: 'gbtree', 'gblinear' or 'dart'.
        [Possible values: {Booster.level_values()}]
        """
    )

    gamma: float = Field(                ########## non-negative acceptable?
        description="""
        Minimum loss reduction required to make a further partition on a leaf node of the tree.
        [Non-negative number] 
        """
    )

    min_child_weight: float = Field(                ########## non-negative acceptable?
        description="""
        Minimum sum of instance weight(hessian) needed in a child.
        [Non-negative number]
        """
    )

    max_delta_step: float = Field(                ########## non-negative acceptable?
        description="""
        Maximum delta step we allow each tree’s weight estimation to be.
        [Non-negative number]
        """
    )

    reg_alpha: float = Field(
        description="""
        L1 regularization term on weights (xgb’s alpha).
        [Positive number]
        """
    )

    reg_lambda: float = Field(                  ########## not sure if this can be set together with reg_alpha
        description="""
        L2 regularization term on weights (xgb’s lambda).
        [Positive number]
        """
    )

XGBClassifierFamily = define_family_type(
    name="XGBClassifierFamily",
    model_type=XGBClassifier,
    hp_config_type=XGBClassifierConfig
)


"""
Other possible parameters:
- grow_policy – Tree growing policy. 0: favor splitting at nodes closest to the node, i.e. grow depth-wise. 1: favor splitting at nodes with highest loss change.
- learning_rate (Optional[float]) – Boosting learning rate (xgb’s “eta”)
- tree_method (Optional[str]) – Specify which tree method to use. Default to auto. If this parameter is set to default, XGBoost will choose the most conservative option available. It’s recommended to study this option from the parameters document tree method
- max_bin – If using histogram-based algorithm, maximum number of bins per feature
- subsample (Optional[float]) – Subsample ratio of the training instance.
- sampling_method – Sampling method. Used only by the GPU version of hist tree method.
    uniform: select random training instances uniformly.
    gradient_based select random training instances with higher probability when the gradient and hessian are larger. (cf. CatBoost)
- colsample_bytree (Optional[float]) – Subsample ratio of columns when constructing each tree.
- colsample_bylevel (Optional[float]) – Subsample ratio of columns for each level.
- colsample_bynode (Optional[float]) – Subsample ratio of columns for each split.
- scale_pos_weight (Optional[float]) – Balancing of positive and negative weights.
- base_score (Optional[float]) – The initial prediction score of all instances, global bias.
- num_parallel_tree (Optional[int]) – Used for boosting random forest.
- eval_metric (Optional[Union[str, List[str], Callable]]) –
    Metric used for monitoring the training result and early stopping. It can be a string or list of strings as names of predefined metric in XGBoost (See doc/parameter.rst), one of the metrics in sklearn.metrics, or any other user defined metric that looks like sklearn.metrics.
    If custom objective is also provided, then custom metric should implement the corresponding reverse link function.
    Unlike the scoring parameter commonly used in scikit-learn, when a callable object is provided, it’s assumed to be a cost function and by default XGBoost will minimize the result during early stopping.
    For advanced usage on Early stopping like directly choosing to maximize instead of minimize, see xgboost.callback.EarlyStopping.
    See Custom Objective and Evaluation Metric for more.
- early_stopping_rounds (Optional[int]) –
    Activates early stopping. Validation metric needs to improve at least once in every early_stopping_rounds round(s) to continue training. Requires at least one item in eval_set in fit().
    If early stopping occurs, the model will have two additional attributes: best_score and best_iteration. These are used by the predict() and apply() methods to determine the optimal number of trees during inference. If users want to access the full model (including trees built after early stopping), they can specify the iteration_range in these inference methods. In addition, other utilities like model plotting can also use the entire model.
    If you prefer to discard the trees after best_iteration, consider using the callback function xgboost.callback.EarlyStopping.
    If there’s more than one item in eval_set, the last entry will be used for early stopping. If there’s more than one metric in eval_metric, the last metric will be used for early stopping.
"""