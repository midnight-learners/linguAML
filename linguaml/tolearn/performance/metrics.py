from typing import Self
from enum import StrEnum

class PerformanceMetric(StrEnum):
    
    ACCURACY = "accuracy"
    F1_SCORE = "f1_score"
    
    @classmethod
    def from_abbreviation(cls, abbreviation: str) -> Self:
        
        abbreviation = abbreviation.lower().strip()
        
        match abbreviation:
            case "accuracy" | "acc":
                return PerformanceMetric.Accuracy
            case "f1_score" | "f1":
                return PerformanceMetric.F1Score
            case _:
                raise ValueError("unknown metric abbreviation")
