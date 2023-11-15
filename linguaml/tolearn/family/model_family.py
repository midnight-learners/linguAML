from pydantic import BaseModel, ConfigDict

# Imports from this package
from linguaml.types import Model
from linguaml.tolearn.hp import HPConfig

class ModelFamily(BaseModel):
    
    hp_config_type: type[HPConfig]
    model_type: type[Model]
    
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        protected_namespaces=()
    )
