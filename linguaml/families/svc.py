from pydantic import Field
from sklearn.svm import SVC
from ..hp import HPConfig, CategoricalHP
from .base import define_family_type

class Kernel(CategoricalHP):
    
    Linear = "linear"
    Poly = "poly"
    RBF = "rbf"
    Sigmoid = "sigmoid"

class DecisionFunctionShape(CategoricalHP):
    
    OVO = "ovo"
    OVR = "ovr"
    
class SVCConfig(HPConfig):
    
    C: float = Field(
        description="""
        Regularization parameter. The strength of the regularization is inversely proportional to C. Must be strictly positive.
        The penalty is a squared l2 penalty.
        [Positive number]
        """
    )
    kernel: Kernel = Field(
        description=f"""
        Specifies the kernel type to be used in the algorithm.
        [Possisble values: {Kernel.level_values()}]
        """
    )
    gamma: float = Field(
        description=f"""
        Kernel coefficient for 'rbf', 'poly' and 'sigmoid'.
        [Positive number]
        """
    )
    tol: float = Field(
        description="""
        Tolerance for stopping criterion.
        [Positive number]
        """
    )
    decision_function_shape: DecisionFunctionShape = Field(
        description=f"""
        Whether to return a one-vs-rest ('ovr') decision function of shape (n_samples, n_classes) as all other classifiers, 
        or the original one-vs-one ('ovo') decision function of libsvm 
        which has shape (n_samples, n_classes * (n_classes - 1) / 2). 
        However, note that internally, one-vs-one ('ovo') is always used as a multi-class strategy to train models; 
        an ovr matrix is only constructed from the ovo matrix. The parameter is ignored for binary classification.
        [Possisble values: {DecisionFunctionShape.level_values()}]
        """
    )

SVCFamily = define_family_type(
    name="SVCFamily",
    model_type=SVC,
    hp_config_type=SVCConfig
)
