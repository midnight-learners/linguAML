# Imports from this package
from linguaml.rl.env import Env
from linguaml.llm.agent import Agent
from linguaml.rl.replay_buffer import ReplayBuffer
from linguaml.tolearn.performance import PerformanceResult, PerformanceResultBuffer
from linguaml.loggers import llm_logger

class LLMTuner:
    
    def __init__(
            self,
            env: Env,
            agent: Agent,
            replay_buffer: ReplayBuffer,
            performance_result_buffer: PerformanceResultBuffer,
        ) -> None:
        
        self._env = env
        self._agent = agent
        self._replay_buffer = replay_buffer
        self._performance_result_buffer = performance_result_buffer
    
    def tune(self, n_epochs: int) -> None:
        """Tune the hyperparameters for n epochs.
        
        Parameters
        ----------
        n_epochs : int
            Number of epochs to tune.
        """
        
        # Reset state
        self._env.reset()
        
        for epoch in range(1, n_epochs + 1):
            self.tune_one_epoch(epoch)
        
    def tune_one_epoch(self, epoch: int = 1) -> None:
        """Tune the hyperparameters for one step.

        Parameters
        ----------
        step : int, optional
            The step number, by default 1.
        """
        
        # Set the epoch number for logging
        llm_logger.configure(extra={"epoch": epoch})
        
        # Ask LLM to generate a new hyperparameter configuration setting
        action = self._agent.select_action(self._performance_result_buffer)
        
        # If the action is None, then stop tuning
        if action is None:
            return
        
        # Interact with the environment
        next_state, reward = self._env.step(action)
        
        # Create a performance result
        performance_result = PerformanceResult(
            hp_config=action.to_hp_config(),
            score=reward if reward is not None else 0.0,
        )
        
        # Logging
        if reward is None:
            llm_logger.warning(f"{performance_result}; Time limit exceeded when fitting the model")
        else:
            llm_logger.info(performance_result)
            
        # Collect the performance result
        self._performance_result_buffer.push(performance_result)
