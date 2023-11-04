import typer
from typing import Optional
from typing_extensions import Annotated
from rich import print
import tomllib
from pathlib import Path
from ..app import app

@app.command()
def train(
        file: Annotated[
            Path,
            typer.Argument(
                help="Configuration file of the training process"
            )
        ],
        log_filepath: Annotated[
            Optional[Path],
            typer.Option(
                "--log", "-l",
                help="File path of the training log to generate"
            )
        ] = None,
    ) -> None:
    
    # Log file will be put under the same parent directory
    # as the training configuration file
    # if it is not specified
    if log_filepath is None:
        log_filepath = file.parent.joinpath(".log")
    
    # Load the training settings from file
    with open(file, "rb") as f:
        training_settings = tomllib.load(f)
    
    # Convert to TrainingSettings instance
    from .config import TrainingSettings
    training_settings = TrainingSettings.model_validate(training_settings)
    
    # Imports
    from torch.optim import Adam
    
    # Imports from this package
    from linguaml.train import train, logger
    from linguaml.env import Env
    from linguaml.data.dataset import load_dataset
    from linguaml.data.replay_buffer import ReplayBuffer
    from linguaml.families import get_family
    from linguaml.agent import Agent
    from linguaml.advantage import AdvantageCalculator
    
    # Set the log file path
    logger.log_filepath = log_filepath
    
    # Environment
    env = Env(
        dataset=load_dataset(name=training_settings.dataset_name),
        family=get_family(name=training_settings.family_name),
        numeric_hp_bounds=training_settings.numeric_hp_bounds,
        state_dim=training_settings.state_dim,
        fitting_time_limit=training_settings.fitting_time_limit,
        random_state=training_settings.random_state
    )
    
    # Agent
    agent = Agent(
        family=get_family(name=training_settings.family_name),
        hidden_size=training_settings.hidden_size,
        cont_dist_family=training_settings.cont_dist_family,
    )
    
    # Advantage calulator
    advantage_calculator = AdvantageCalculator(
        moving_average_alg=training_settings.moving_average_alg,
        period=training_settings.sma_period,
        alpha=training_settings.ema_alpha
    )
    
    # Train!
    train(
        env=env,
        agent=agent,
        optimizer=Adam(
            agent.parameters(), 
            lr=training_settings.adam_lr
        ),
        n_epochs=training_settings.n_epochs,
        replay_buffer=ReplayBuffer(
            capacity=training_settings.replay_buffer_capacity
        ),
        n_timesteps_per_episode=training_settings.n_timesteps_per_episode,
        advantage_calculator=advantage_calculator,
        batch_size=training_settings.batch_size,
        n_epochs_for_updating_agent=training_settings.n_epochs_for_updating_agent,
        ppo_epsilon=training_settings.ppo_epsilon
    )
    
    