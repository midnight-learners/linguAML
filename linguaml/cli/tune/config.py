from typing import Optional
from pydantic import BaseModel, ConfigDict

# Imports from this package
from linguaml.tolearn.hp.bounds import NumericHPBounds

class TuningSettings(BaseModel):
    
    dataset_names: list[str]
    family_name: str
    random_state: Optional[int] = None
    n_epochs: int
    lookback: int 
    fitting_time_limit: float
    hidden_size: int
    cont_dist_family: str
    adam_lr: float
    moving_average_alg: str
    sma_period: int
    ema_alpha: float
    replay_buffer_capacity: int
    n_timesteps_per_episode: int
    n_steps_for_updating_agent: int
    batch_size: int
    min_batch_size: Optional[int] = None
    ppo_epsilon: float
    numeric_hp_bounds: NumericHPBounds
    
    model_config = ConfigDict(
        frozen=True,
        extra="ignore"
    )
