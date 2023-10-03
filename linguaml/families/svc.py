from sklearn.svm import SVC
from ..hp import HPConfig, CategoricalHP
from .base import define_family_type

class SVCKernel(CategoricalHP):
    
    Linear = "linear"
    Poly = "poly"
    RBF = "rbf"
    Sigmoid = "sigmoid"

class SVCConfig(HPConfig):
    
    C: float
    kernel: SVCKernel
    gamma: float
    tol: float

SVCFamily = define_family_type(
    name="SVCFamily",
    model_type=SVC,
    hp_config_type=SVCConfig
)
