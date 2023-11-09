# Imports from this package
from .schemas import ChatModel, SystemMessage, UserMessage
from .utils import get_hp_config_description, get_hp_output_format
from linguaml.families.base import Family
from linguaml.data.performance_result_buffer import PerformanceResultBuffer
from linguaml.hp import HPConfig

prompt_template = """
You are fine tuning a {model_name} model.
Below are some of the high-performance hyperparameter configuration settings that you have tried, \
along with their corresponding accuracy levels on the validation dataset:
{performance_results_str}

Please strictly refer to the following hyperparameter configuration specification for the {model_name}
(*IMPORTANT*: the domain of each hyperparameter is enclosed by []):
{hp_config_description}

Please choose a new hyperparameter configuration setting to possibly achieve higher accuracy.
(*IMPORTANT*: The hyperparameter configuration setting you choose must be different from what you have already tried!)

Your output should be in the valid JSON format of:
{output_format}

Your output is (in JSON format):
"""

class LLMAgent:
    
    def __init__(
            self, 
            family: Family,
            chat_model: ChatModel,
        ) -> None:
        
        self._family = family
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
        ) -> HPConfig:
        """Select a hyperparameter configuration setting 
        based on the historical performance results.
        
        Parameters
        ----------
        performance_result_buffer : PerformanceResultBuffer
            A buffer of performance results.
            
        Returns
        -------
        HPConfig
            The hyperparameter configuration setting.
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
        message = self._chat_model.invoke([
            SystemMessage("You are a professional data scientist."),
            UserMessage(prompt)
        ])
        hp_config = self._family.hp().model_validate_json(message.content)
        
        return hp_config
