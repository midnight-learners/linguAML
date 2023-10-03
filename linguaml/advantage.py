from typing import Self
from enum import Enum, auto
from collections import deque
import numpy as np

class MovingAverageAlg(Enum):
    
    SimpleMovingAverage = auto()
    ExponentialMovingAverage = auto()
    
    @classmethod
    def from_abbreviation(cls, abbreviation: str) -> Self:
        
        abbreviation = abbreviation.lower().strip()
        
        match abbreviation:
            case "sma" | "ma":
                return MovingAverageAlg.SimpleMovingAverage
            case "ema" | "ewma":
                return MovingAverageAlg.ExponentialMovingAverage
            case _:
                raise ValueError("unknown algorithm abbreviation")

class AdvantageCalculator:
    
    def __init__(
            self,
            *,
            moving_average_alg: MovingAverageAlg | str = MovingAverageAlg.SimpleMovingAverage,
            period: int = 5,
            alpha: float = 0.2,
        ) -> None:
        
        if isinstance(moving_average_alg, str):
            moving_average_alg = MovingAverageAlg.from_abbreviation(moving_average_alg)
        self._moving_average_alg = moving_average_alg
        
        self._period = period
        self._alpha = alpha
        
        self._reward_buffer: deque[float] = deque(maxlen=period)
        self._reward_buffer.extend([0.0] * period)
        self._last_moving_average_value = 0.0
        
    def __call__(self, reward: float) -> float:
        
        return self.calculate_advantage(reward)
        
    def calculate_advantage(self, reward: float) -> float:
        
        baseline = self.calculate_baseline(reward)
        advantage = reward - baseline
        
        return advantage
        
    def calculate_baseline(self, reward: float) -> float:
        
        match self._moving_average_alg:
            
            case MovingAverageAlg.SimpleMovingAverage:
                
                baseline = np.mean(self._reward_buffer)
                
            case MovingAverageAlg.ExponentialMovingAverage:
                
                baseline = self._last_moving_average_value
                
                self._last_moving_average_value = reward * self._alpha\
                    + self._last_moving_average_value * (1 - self._alpha)
                
        self._reward_buffer.append(reward)
        
        return baseline
