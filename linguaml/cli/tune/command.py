import typer
from typing import Optional, Annotated
from rich import print
import tomllib
from pathlib import Path

# Imports from this package
from ..app import app
from linguaml.types import TunerRole

@app.command()
def tune(
        files: Annotated[
            list[Path],
            typer.Argument(
                help="Configuration files of the tuning process."
            )
        ],
        log_filepath: Annotated[
            Optional[Path],
            typer.Option(
                "--log", "-l",
                help="File path of the tuning log to generate."
            )
        ] = None,
        role: Annotated[
            TunerRole,
            typer.Option(
                "--role", "-r",
                help="Role of the tuner."
            )
        ] = TunerRole.RL
    ) -> None:
    
    # Log file will be put under the same parent directory
    # as the tuning configuration file
    # if it is not specified
    if log_filepath is None:
        # Check all tuning configuration files share the same parent
        parents = set([file.parent for file in files])
        assert len(parents) == 1,\
            "All configuration files must be placed under the same directory"
        parent = parents.pop()
        
        # Set the log file path
        log_filepath = parent.joinpath(".log")
    
    # Load the training settings from files
    tuning_settings = {}
    for file in files:
        with open(file, "rb") as f:
            more_tuning_settings = tomllib.load(f)
        tuning_settings.update(more_tuning_settings)
    
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
