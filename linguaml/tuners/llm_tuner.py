# Imports from this package
from linguaml.rl.env import Env
from linguaml.llm import LLMAgent
from linguaml.rl.replay_buffer import ReplayBuffer
from linguaml.tolearn.performance import PerformanceResult, PerformanceResultBuffer
from linguaml.logger import logger

logger = logger.bind(tuner_role="llm")

class LLMTuner:
    
    def __init__(
            self,
            env: Env,
            agent: LLMAgent,
            replay_buffer: ReplayBuffer,
            performance_result_buffer: PerformanceResultBuffer,
        ) -> None:
        
        self._env = env
        self._agent = agent
        self._replay_buffer = replay_buffer
        self._performance_result_buffer = performance_result_buffer
    
    def tune(self, n_epoch: int, start_epoch: int = 1) -> None:
        """Tune the hyperparameters for n epochs.
        
        Parameters
        ----------
        n_epoch : int
            Number of epochs to tune.
        start_epoch : int, optional
            The starting epoch number, by default 1.
        """
        
        for epoch in range(start_epoch, n_epoch + 1):
            self.tune_one_epoch(epoch)
        
    def tune_one_epoch(self, epoch: int = 1) -> None:
        """Tune the hyperparameters for one step.

        Parameters
        ----------
        step : int, optional
            The step number, by default 1.
        """
        
        # Set the epoch number for logging
        logger.configure(extra={"epoch": epoch})
        
        # Ask LLM to generate a new hyperparameter configuration setting
        hp_config = self._agent.select_hp_cnofig(self._performance_result_buffer)
        
        # Interact with the environment
        next_state, reward = self._env.step(hp_config=hp_config)
        
        # Create a performance result
        performance_result = PerformanceResult(
            hp_config=hp_config,
            accuracy=reward if reward is not None else 0.0,
        )
        
        # Logging
        if reward is None:
            logger.warning(f"{performance_result}; Time limit exceeded when fitting the model")
        else:
            logger.info(performance_result)
            
        # Collect the performance result
        self._performance_result_buffer.push(performance_result)
