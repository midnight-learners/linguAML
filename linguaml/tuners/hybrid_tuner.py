import numpy as np
import torch
from torch.optim import Optimizer

# Imports from this package
from linguaml.rl.env import Env
from linguaml.rl.agent import Agent
from linguaml.rl.action import Action
from linguaml.rl.transition import Transition, BatchedTransitions
from linguaml.rl.replay_buffer import ReplayBuffer, ReplayBufferLoader
from linguaml.rl.advantage import AdvantageCalculator
from linguaml.rl.loss import ppo_loss
from linguaml.tolearn.performance import PerformanceResult
from linguaml.logger import logger

logger = logger.bind(tuner_role="hybrid")

class HybridTuner:
    
    def __init__(
            self,
            env: Env, 
            agent: Agent,
            replay_buffer: ReplayBuffer,
            advantage_calculator: AdvantageCalculator,
        ) -> None:
        
        self._env = env
        self._agent = agent
        self._replay_buffer = replay_buffer
        self._advantage_calculator = advantage_calculator
    
    def tune():
        
        pass