from pydantic import Field
from sklearn.linear_model import LogisticRegression
from functools import partial
from ..hp import HPConfig, CategoricalHP
from .base import define_family_type

class Penalty(CategoricalHP):
    
    NONE = "none"
    L1 = "l1"
    L2 = "l2"
    Elasticnet = "elasticnet"
    
class LogisticRegressionConfig(HPConfig):
    
    penalty: Penalty = Field(
        description=f"""
        Specify the norm of the penalty:
        
        'none': no penalty is added;
        'l2': add a L2 penalty term and it is the default choice;
        'l1': add a L1 penalty term;
        'elasticnet': both L1 and L2 penalty terms are added.
        
        [Possible values: {Penalty.level_values()}]
        """
    )
    
    tol: float = Field(
        description="""
        Tolerance for stopping criterion.
        [Positive number]
        """
    )
    
    C: float = Field(
        description="""
        Regularization parameter. The strength of the regularization is inversely proportional to C. Must be strictly positive.
        The penalty is a squared l2 penalty.
        [Positive number]
        """
    )
    
    max_iter: int = Field(
        description="""
        Maximum number of iterations taken for the solvers to converge.
        [Positive integer]
        """
    )
    
    l1_ratio: float = Field(
        description="""
        The Elastic-Net mixing parameter, with 0 <= l1_ratio <= 1. Only used if penalty='elasticnet'. Setting l1_ratio=0 is equivalent to using penalty='l2', while setting l1_ratio=1 is equivalent to using penalty='l1'. For 0 < l1_ratio <1, the penalty is a combination of L1 and L2.
        [A real number in between 0 and 1]
        """
    )    

LogisticRegressionFamily = define_family_type(
    name="LogisticRegressionFamily",
    model_type=partial(LogisticRegression, solver="saga"),
    hp_config_type=LogisticRegressionConfig
)
