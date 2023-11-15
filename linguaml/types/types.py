from typing import TypeAlias
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin

Number: TypeAlias = float | int
NumberList: TypeAlias = list[Number]
Model: TypeAlias = BaseEstimator | ClassifierMixin | RegressorMixin
