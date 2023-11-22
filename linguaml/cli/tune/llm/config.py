from typing import Optional
from pydantic import BaseModel, ConfigDict

# Imports from this package
from linguaml.tolearn.hp.bounds import NumericHPBounds
from linguaml.llm.openai.chat import OpenAIChatModelName

class TuningSettings(BaseModel):
    
    dataset_names: list[str]
    family_name: str
    random_state: Optional[int] = None
    n_epochs: int
    performance_metric: str
    lookback: int
    fitting_time_limit: float
    replay_buffer_capacity: int
    performance_result_buffer_capacity: int
    numeric_hp_bounds: NumericHPBounds
    chat_model_name: OpenAIChatModelName
    temperature: float
    
    model_config = ConfigDict(
        frozen=True,
        extra="ignore"
    )
