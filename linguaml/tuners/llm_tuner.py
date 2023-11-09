# Imports from this package
from linguaml.logger import logger
from linguaml.env import Env
from linguaml.llm import LLMAgent
from linguaml.data.performance_result_buffer import PerformanceResult, PerformanceResultBuffer

def tune(
        env: Env,
        agent: LLMAgent,
        n_epochs: int,
        performance_result_buffer: PerformanceResultBuffer,
    ):
    
    for epoch in range(n_epochs):
    
        # Set the epoch number for logging
        logger.extra["epoch"] = epoch + 1
        
        # Ask LLM to generate a new hyperparameter configuration setting
        hp_config = agent.select_hp_cnofig(performance_result_buffer)
        
        # Interact with the environment
        next_state, reward = env.step(hp_config=hp_config)
        
        # Logging
        message_parts = [
            f"Epoch: {logger.extra['epoch']}",
            f"Hyperparameters: {hp_config}"
        ]

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
        performance_result_buffer.push(
            PerformanceResult(
                hp_config=hp_config,
                accuracy=reward if reward is not None else 0.0,
            )
        )
