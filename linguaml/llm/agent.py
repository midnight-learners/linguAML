from typing import Optional

# Imports from this package
from .schemas import ChatModel, SystemMessage, UserMessage
from .utils import get_hp_config_description, get_hp_output_format
from linguaml.tolearn.family import Family
from linguaml.tolearn.hp.bounds import NumericHPBounds
from linguaml.tolearn.performance import PerformanceResultBuffer
from linguaml.tolearn.hp import HPConfig
from linguaml.rl.action import ActionConfig, Action
from linguaml.loggers import llm_logger

prompt_template = """
You are fine tuning a {model_name} model.
Below are some of the high-performance hyperparameter configuration settings that you have tried, \
along with their corresponding score levels on the validation dataset:
{performance_results_str}

Please strictly refer to the following hyperparameter configuration specification for the {model_name}
(*IMPORTANT*: the domain of each hyperparameter is enclosed by []):
{hp_config_description}

Please choose a new hyperparameter configuration setting to possibly achieve higher score.
(*IMPORTANT*: The hyperparameter configuration setting you choose must be different from what you have already tried!)

Your output should be in the valid JSON format of:
{output_format}

Your output is (in JSON format):
"""

class Agent:
    
    def __init__(
            self, 
            family: Family,
            numeric_hp_bounds: NumericHPBounds,
            chat_model: ChatModel,
        ) -> None:
        
        # Configure the action
        ActionConfig.family = family
        ActionConfig.numeric_hp_bounds = numeric_hp_bounds
        
        # Model family
        self._family = family
        
        # Chat model
        self._chat_model = chat_model
        
        # Get the description of the hyperparameter configuration
        self._hp_config_description = get_hp_config_description(family)
        
        # Get the output format of the hyperparameter configuration
        self._output_format = get_hp_output_format(family)
        
    @property
    def family(self) -> Family:
        """The model family that this agent is fine tuning.
        """
        
        return self._family
        
    def select_hp_cnofig(
            self, 
            performance_result_buffer: PerformanceResultBuffer
        ) -> Optional[HPConfig]:
        """Select a hyperparameter configuration setting 
        based on the historical performance results.
        
        Parameters
        ----------
        performance_result_buffer : PerformanceResultBuffer
            A buffer of performance results.
            
        Returns
        -------
        Optional[HPConfig]
            A hyperparameter configuration setting.
            Return None if failed to select a hyperparameter configuration setting.
        """
        
        # Make prompt
        prompt = prompt_template.format(
            model_name=self._family.name(),
            performance_results_str="\n".join(list(map(
                lambda result: str(result), 
                performance_result_buffer.peek_first_n_high_performance_results(5)
            ))),
            hp_config_description=self._hp_config_description,
            output_format=self._output_format
        )
        
        # Ask LLM to generate a new hyperparameter configuration setting
        try:
            message = self._chat_model.invoke([
                SystemMessage("You are a professional data scientist."),
                UserMessage(prompt)
            ])
        except:
            llm_logger.error("Failed to get responses from the LLM")
            return None
        
        # Parse the hyperparameter configuration setting from LLM's output
        try:
            hp_config = self._family.hp().model_validate_json(message.content)
        except:
            llm_logger.error("Failed to parse the hyperparameter configuration setting from LLM's output")
            return None
        
        return hp_config

    def select_action(self, performance_result_buffer: PerformanceResultBuffer) -> Optional[Action]:
        
        # Select a hyperparameter configuration
        hp_config = self.select_hp_cnofig(performance_result_buffer)
        
        # Return None if failed to select a hyperparameter configuration setting
        if hp_config is None:
            return None
        
        # Convert to an action
        action = Action.from_hp_config(hp_config)
        
        return action
