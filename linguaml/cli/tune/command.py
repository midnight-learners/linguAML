import typer
from typing import Optional
from typing_extensions import Annotated
from rich import print
import tomllib
from pathlib import Path

from ..app import app

@app.command()
def tune(
        file: Annotated[
            Path,
            typer.Argument(
                help="Configuration file of the tuning process"
            )
        ],
        log_filepath: Annotated[
            Optional[Path],
            typer.Option(
                "--log", "-l",
                help="File path of the tuning log to generate"
            )
        ] = None,
    ) -> None:
    
    # Log file will be put under the same parent directory
    # as the tuning configuration file
    # if it is not specified
    if log_filepath is None:
        log_filepath = file.parent.joinpath(".log")
    
    # Load the training settings from file
    with open(file, "rb") as f:
        tuning_settings = tomllib.load(f)
    
    # Convert to TrainingSettings instance
    from .config import TuningSettings
    tuning_settings = TuningSettings.model_validate(tuning_settings)
    
    # Imports
    from torch.optim import Adam
    
    # Imports from this package
    from linguaml.data.dataset import load_dataset
    from linguaml.tuners import RLTuner
    from linguaml.rl.agent import Agent
    from linguaml.rl.env import Env
    from linguaml.rl.replay_buffer import ReplayBuffer
    from linguaml.tolearn.family import Family
    from linguaml.rl.advantage import AdvantageCalculator
    
    # Set the log file path
    from linguaml.logger import set_log_filepath
    from linguaml.tuners.rl_tuner import logger
    set_log_filepath(logger, log_filepath)
    
    # Environment
    env = Env(
        datasets=[
            load_dataset(name=name)
            for name in tuning_settings.dataset_names
        ],
        lookback=tuning_settings.lookback,
        fitting_time_limit=tuning_settings.fitting_time_limit,
        random_state=tuning_settings.random_state
    )
    
    # Agent
    agent = Agent(
        family=Family.from_name(tuning_settings.family_name),
        numeric_hp_bounds=tuning_settings.numeric_hp_bounds,
        hidden_size=tuning_settings.hidden_size,
        cont_dist_family=tuning_settings.cont_dist_family,
    )
    
    # Advantage calulator
    advantage_calculator = AdvantageCalculator(
        moving_average_alg=tuning_settings.moving_average_alg,
        period=tuning_settings.sma_period,
        alpha=tuning_settings.ema_alpha
    )
    
    # Create a tuner
    tuner = RLTuner(
        env=env,
        agent=agent,
        replay_buffer=ReplayBuffer(
            capacity=tuning_settings.replay_buffer_capacity
        ),
        advantage_calculator=advantage_calculator
    )
    
    # Tune!
    tuner.tune(
        n_epochs=tuning_settings.n_epochs,
        n_timesteps_per_episode=tuning_settings.n_timesteps_per_episode,
        batch_size=tuning_settings.batch_size,
        n_steps_for_updating_agent=tuning_settings.n_steps_for_updating_agent,
        optimizer=Adam(
            agent.parameters(), 
            lr=tuning_settings.adam_lr
        ),
        ppo_epsilon=tuning_settings.ppo_epsilon
    )
