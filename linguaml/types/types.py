from typing import TypeAlias
from enum import StrEnum
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin

Number: TypeAlias = float | int
NumberList: TypeAlias = list[Number]
Model: TypeAlias = BaseEstimator | ClassifierMixin | RegressorMixin

class TunerRole(StrEnum):
    
    RL = "rl"
    LLM = "llm"
    HYBRID = "hybrid"
