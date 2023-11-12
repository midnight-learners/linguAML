from typing import Optional
from pydantic import BaseModel

# Imports from this package
from linguaml.hp import HPConfig


class PerformanceResult(BaseModel):
    
    hp_config: HPConfig
    accuracy: float