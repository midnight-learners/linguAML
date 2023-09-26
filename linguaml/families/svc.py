from sklearn.svm import SVC
from ..hp import HpConfig
from .base import define_family_type

class SVCConfig(HpConfig):
    
    C: float
    gamma: float
    tol: float
    
SVCFamily = define_family_type(
    name="SVCFamily",
    model_type=SVC,
    hp_config_type=SVCConfig
)
