# Imports from this package
from linguaml.logger import logger
from linguaml.env import Env
from linguaml.llm import LLMAgent
from linguaml.data.replay_buffer import ReplayBuffer
from linguaml.data.performance_result_buffer import PerformanceResult, PerformanceResultBuffer

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
    
    def tune(self, n_steps: int, start_step: int = 1) -> None:
        """Tune the hyperparameters for n steps.

        Parameters
        ----------
        n_steps : int
            The number of steps to tune the hyperparameters.
        start_step : int, optional
            The step number to start tuning, by default 1.
        """
        
        for step in range(start_step, n_steps + 1):
            self.tune_one_step(step=step)
        
    def tune_one_step(self, step: int = 1) -> None:
        """Tune the hyperparameters for one step.

        Parameters
        ----------
        step : int, optional
            The step number, by default 1.
        """
        
        # Set the epoch number for logging
        logger.extra["step"] = step
        
        # Ask LLM to generate a new hyperparameter configuration setting
        hp_config = self._agent.select_hp_cnofig(self._performance_result_buffer)
        
        # Interact with the environment
        next_state, reward = self._env.step(hp_config=hp_config)
        
        # Logging
        message_parts = [
            f"Step: {logger.extra['step']}",
            f"Hyperparameters: {hp_config}"
        ]

        # Log the performance result
        # Warn the user if the model fitting exceeds the time limit
        if reward is None:
            message_parts.extend([
                f"Accuracy: {reward}",
                "Model fitting exceeds time limit"
            ])
            message = "; ".join(message_parts)
            logger.warning(message)
        
        # Log the performance result
        else:
            message_parts.append(f"Accuracy: {reward}")
            message = "; ".join(message_parts)
            logger.info(message)
            
        # Collect the performance result
        self._performance_result_buffer.push(
            PerformanceResult(
                hp_config=hp_config,
                accuracy=reward if reward is not None else 0.0,
            )
        )
