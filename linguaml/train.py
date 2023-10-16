import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from .logger import logger
from .data.transition import Transition, convert_to_transition_with_fields_as_lists
from .data.replay_buffer import ReplayBuffer
from .env import Env
from .agent import Agent
from .advantage import AdvantageCalculator
from .action import convert_action_to_hp_config

def train(
        env: Env,
        agent: Agent,
        optimizer: Optimizer,
        n_epochs: int,
        replay_buffer: ReplayBuffer,
        n_timesteps_per_episode: int,
        advantage_calculator: AdvantageCalculator,
        batch_size: int,
        n_epochs_for_updating_agent: int,
        ppo_epsilon: float
    ):
        
    for epoch in range(n_epochs):
        
        # Update logger's epoch
        logger.extra["epoch"] = epoch + 1
        
        # Collect transitions
        sample_transitions(
            replay_buffer=replay_buffer,
            env=env,
            agent=agent,
            n_timesteps_per_episode=n_timesteps_per_episode,
            advantage_calculator=advantage_calculator
        )
        
        # Average reward of all sample in the buffer
        rewards = convert_to_transition_with_fields_as_lists(replay_buffer).reward
        avg_reward = np.mean(rewards)
        logger.info(
            (
                "Epoch: {epoch}; "
                "Average Sample Reward: {avg_reward}"
            ).format(
                epoch=logger.extra["epoch"],
                avg_reward=avg_reward
            )
        )
        
        # Create a data loader
        replay_buffer_loader = DataLoader(
            replay_buffer,
            batch_size=batch_size
        )
        
        # Train the agent
        update_agent(
            agent=agent,
            optimizer=optimizer,
            replay_buffer_loader=replay_buffer_loader,
            n_epochs=n_epochs_for_updating_agent,
            ppo_epsilon=ppo_epsilon
        )
        
        # Clear replay buffer
        replay_buffer.clear()

def sample_transitions(
        replay_buffer: ReplayBuffer,
        env: Env,
        agent: Agent,
        n_timesteps_per_episode: int,
        advantage_calculator: AdvantageCalculator
    ) -> None:
        
    while len(replay_buffer) < replay_buffer.capacity:
        
        # Collect transitions by interacting with the env
        transitions = play_one_episode(
            env=env,
            agent=agent,
            n_timesteps_per_episode=n_timesteps_per_episode,
            advantage_calculator=advantage_calculator
        )
        
        # Add to the buffer
        replay_buffer.extend(transitions)

def play_one_episode(
        env: Env,
        agent: Agent,
        n_timesteps_per_episode: int,
        advantage_calculator: AdvantageCalculator
    ) -> list[Transition]:
    
    transitions = []
    state = env.reset()
    
    for t in range(n_timesteps_per_episode):
        
        # Select an action
        state = torch.tensor(state, dtype=torch.float)
        action = agent.select_action(state)
        action = action.detach().numpy()
        
        # Compute the log-probability of the action taken
        log_prob = agent.log_prob()
        log_prob = log_prob.detach().item()
        
        # Interact with the env
        next_state, reward = env.step(action)
        
        # Logging
        hp_config = convert_action_to_hp_config(
            action=action,
            family=env.family,
            numeric_hp_bounds=env.numeric_hp_bounds
        )
        
        message_parts = [
            f"Epoch: {logger.extra['epoch']}",
            f"Hyperparameters: {hp_config}"
        ]
                
        if reward is None:
            # The reward should be set zero
            reward = 0.0
            
            message_parts.extend([
                f"Accuracy: {reward}",
                "Model fitting exceeds time limit"
            ])
            
            message = "; ".join(message_parts)
            
            # Warning log
            logger.warning(message)
        
        
        else:
            message_parts.append(f"Accuracy: {reward}")
            message = "; ".join(message_parts)
            logger.info(message)
            
        # Compute advantage using moving average technique
        advantage = advantage_calculator(reward)
            
        # Transition sample
        transition = Transition(
            state=state,
            action=action,
            reward=reward,
            advantage=advantage,
            log_prob=log_prob
        )
        transitions.append(transition)

        # Step to the next state
        state = next_state

    return transitions

def update_agent(
        agent: Agent,
        optimizer: Optimizer,
        replay_buffer_loader: DataLoader,
        n_epochs: int,
        ppo_epsilon: float = 0.2
    ):
    
    for _ in range(n_epochs):
        
        transition: Transition
        for transition in replay_buffer_loader:
            
            # Compute PPO loss
            loss = ppo_loss(
                curr_log_prob=agent.log_prob(
                    transition.action,
                    transition.state
                ),
                old_log_prob=transition.log_prob,
                advantage=transition.advantage,
                epsilon=ppo_epsilon
            )
            
            # Update the agent
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

def ppo_loss(
        *,
        curr_log_prob: Tensor,
        old_log_prob: Tensor,
        advantage: Tensor,
        epsilon: float = 0.2
    ) -> Tensor:
    
    ratio = torch.exp(curr_log_prob - old_log_prob)
    
    surr1 = ratio * advantage
    surr2 = torch.clip(
        ratio,
        1 - epsilon,
        1 + epsilon
    ) * advantage
    
    loss = -torch.min(surr1, surr2).mean()
    
    return loss

def sma(x: np.ndarray, period: int = 5) -> np.ndarray:
    """Simple Moving Average. 
    Compute the simple moving agverage of the input time series. 

    Parameters
    ----------
    x : np.ndarray
        _description_
    period : int, optional
        _description_, by default 5

    Returns
    -------
    np.ndarray
        _description_
    """
    
    assert len(x) >= period,\
        "the length of the array must be at least the length of the period"
    
    sma = []
    for t in range(period - 1, len(x)):
        sma.append(x[t-period+1:t].mean(axis=0))

    return np.array(sma)

def ema(
        x: np.ndarray, 
        period: int = 5,
        alpha: float = 0.2
    ) -> np.ndarray:
    """Exponential Moving Average. 
    Compute the exponential moving agverage of the input time series. 


    Parameters
    ----------
    x : np.ndarray
        _description_
    period : int, optional
        _description_, by default 5
    alpha : float, optional
        _description_, by default 0.2

    Returns
    -------
    np.ndarray
        _description_
    """
    
    assert len(x) >= period,\
        "the length of the array must be at least the length of the period"
    
    ema = []
    ema.append(x[:period].mean(axis=0))
    for i, t in enumerate(range(period, len(x))):
        ema.append(
            x[t] * alpha + ema[i] * (1 - alpha)
        )

    return np.array(ema)
