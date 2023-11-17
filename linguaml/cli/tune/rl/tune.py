from .config import TuningSettings

def tune(tuning_settings: TuningSettings) -> None:

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
