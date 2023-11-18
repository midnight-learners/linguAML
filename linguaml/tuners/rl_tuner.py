from typing import Optional
import numpy as np
import torch
from torch.optim import Optimizer

# Imports from this package
from linguaml.rl.env import Env
from linguaml.rl.agent import Agent
from linguaml.rl.action import Action
from linguaml.rl.transition import Transition, BatchedTransitions
from linguaml.rl.replay_buffer import ReplayBuffer, ReplayBufferLoader
from linguaml.rl.advantage import AdvantageCalculator
from linguaml.rl.loss import ppo_loss
from linguaml.tolearn.performance import PerformanceResult
from linguaml.loggers import rl_logger

class RLTuner:
    
    def __init__(
            self,
            env: Env, 
            agent: Agent,
            replay_buffer: ReplayBuffer,
            advantage_calculator: AdvantageCalculator,
        ) -> None:
        
        self._env = env
        self._agent = agent
        self._replay_buffer = replay_buffer
        self._advantage_calculator = advantage_calculator
    
    def tune(
            self,
            n_epochs: int,
            n_timesteps_per_episode: int,
            batch_size: int,
            n_steps_for_updating_agent: int,
            optimizer: Optimizer,
            ppo_epsilon: float,
            min_batch_size: Optional[int] = None
        ):
        
        for epoch in range(n_epochs):
            
            # Set the epoch by updating logger's extra
            rl_logger.configure(extra={"epoch": epoch + 1})
            
            # Collect transitions
            self.sample_transitions(n_timesteps_per_episode=n_timesteps_per_episode)
            
            # Average reward of all sample in the buffer
            batched_transitions = BatchedTransitions.from_transitions(self._replay_buffer)
            batched_rewards = batched_transitions.reward
            avg_reward = np.mean(batched_rewards)
            rl_logger.info(f"Average sample reward: {avg_reward}")
            
            # Create a data loader that generates batched transitions
            replay_buffer_loader = ReplayBufferLoader(
                self._replay_buffer,
                batch_size=batch_size,
                min_batch_size=min_batch_size
            )
            
            # Train the agent
            self.update_agent(
                optimizer=optimizer,
                replay_buffer_loader=replay_buffer_loader,
                n_steps_for_updating_agent=n_steps_for_updating_agent,
                ppo_epsilon=ppo_epsilon
            )
            
            # Clear replay buffer
            self._replay_buffer.clear()
    
    def sample_transitions(
            self,
            n_timesteps_per_episode: int,
        ) -> None:
            
        while len(self._replay_buffer) < self._replay_buffer.capacity:
            
            # Collect transitions by interacting with the env
            transitions = self.play_one_episode(n_timesteps_per_episode)
            
            # Add to the buffer
            self._replay_buffer.extend(transitions)

    def play_one_episode(
            self,
            n_timesteps_per_episode: int,
        ) -> list[Transition]:
        
        transitions = []
        state = self._env.reset()
        
        for t in range(n_timesteps_per_episode):
            
            # Select an action
            action: Action = self._agent.select_action(state)
            
            # Compute the log-probability of the action taken
            log_prob = self._agent.log_prob(action)
            
            # Interact with the environment
            next_state, reward = self._env.step(action)
            
            # Log the performance result
            performance_result = PerformanceResult(
                hp_config=action.to_hp_config(),
                accuracy=reward if reward is not None else 0.0
            )
            if reward is None:
                # Wrarning log
                rl_logger.warning(f"{performance_result}; Time limit exceeded when fitting the model")
                
                # Set the reward to 0
                reward = 0.0
            else:
                rl_logger.info(performance_result)
            
            # Compute advantage using moving average algorithm
            advantage = self._advantage_calculator(reward)
                
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
            self,
            optimizer: Optimizer,
            replay_buffer_loader: ReplayBufferLoader,
            n_steps_for_updating_agent: int,
            ppo_epsilon: float = 0.2
        ) -> None:
        
        for _ in range(n_steps_for_updating_agent):
            
            for batched_transitions in replay_buffer_loader.gen_batches():
                
                # Compute PPO loss
                loss = ppo_loss(
                    curr_log_prob=self._agent.log_prob(
                        batched_transitions.action,
                        batched_transitions.state
                    ),
                    old_log_prob=torch.tensor(batched_transitions.log_prob),
                    advantage=torch.tensor(batched_transitions.advantage),
                    epsilon=ppo_epsilon
                )
                
                # Update the agent
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
