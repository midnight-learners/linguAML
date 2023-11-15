from pydantic import Field
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression

# Imports from this package
from ..hp import HPConfig, CategoricalHP
from .model_family import ModelFamily

class Penalty(CategoricalHP):
    
    NONE = "none"
    L1 = "l1"
    L2 = "l2"
    ELASTICNET = "elasticnet"
    
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

class ModifiedLogisticRegression(ClassifierMixin, BaseEstimator):
    
    def __init__(
            self,
            *,
            penalty: str = "l2",
            tol: float = 1e-4,
            C: float = 1.0,
            max_iter: int = 100,
            l1_ratio: float = 0.5,
            solver: str = "saga"
        ):
        
        self.penalty = penalty
        self.tol = tol
        self.C = C
        self.max_iter = max_iter
        self.l1_ratio = l1_ratio
        self.solver = solver
        
        self._model = LogisticRegression(
            penalty=penalty,
            tol=tol,
            C=C,
            max_iter=max_iter,
            l1_ratio=l1_ratio,
            solver=solver
        )
    
    def fit(self, X, y):
        
        self._model.fit(X, y)
        
        return self
    
    def predict(self, X):
        
        return self._model.predict(X)
    
    def predict_proba(self, X):
        
        return self._model.predict_proba(X)

logistic_regression_family = ModelFamily(
    hp_config_type=LogisticRegressionConfig,
    model_type=ModifiedLogisticRegression,
)
