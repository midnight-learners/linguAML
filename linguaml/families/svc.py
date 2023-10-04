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
    
    C: float
    kernel: Kernel
    gamma: float
    tol: float
    decision_function_shape: DecisionFunctionShape

SVCFamily = define_family_type(
    name="SVCFamily",
    model_type=SVC,
    hp_config_type=SVCConfig
)
