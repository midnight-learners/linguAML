from typing import Optional
import random
import numpy as np
import torch
from torch.optim import Optimizer

# Imports from this package
from linguaml.rl.env import Env
from linguaml.rl.agent import Agent as RLAgent
from linguaml.rl.action import Action
from linguaml.rl.state import State
from linguaml.rl.transition import Transition, BatchedTransitions
from linguaml.rl.replay_buffer import ReplayBuffer, ReplayBufferLoader
from linguaml.rl.advantage import AdvantageCalculator
from linguaml.rl.loss import ppo_loss
from linguaml.llm.agent import Agent as LLMAgent
from linguaml.tolearn.performance import PerformanceResult, PerformanceResultBuffer
from linguaml.loggers import rl_logger, llm_logger, hybrid_logger

class HybridTuner:
    
    def __init__(
            self,
            *,
            env: Env, 
            rl_agent: RLAgent,
            llm_agent: LLMAgent,
            replay_buffer: ReplayBuffer,
            performance_result_buffer: PerformanceResultBuffer,
            advantage_calculator: AdvantageCalculator,
            llm_agent_sampling_freq: float
        ) -> None:
        
        self._env = env
        self._rl_agent = rl_agent
        self._llm_agent = llm_agent
        self._replay_buffer = replay_buffer
        self._performance_result_buffer = performance_result_buffer
        self._advantage_calculator = advantage_calculator
        self._llm_agent_sampling_freq = llm_agent_sampling_freq
        
        # Keep track of the current state
        self._state = self._env.reset()
    
    def tune(
            self,
            n_epochs: int,
            batch_size: int,
            n_steps_for_updating_agent: int,
            optimizer: Optimizer,
            ppo_epsilon: float,
            min_batch_size: Optional[int] = None
        ) -> None:
        
        for epoch in range(n_epochs):
            
            # Set the epoch by updating loggers' extra
            rl_logger.configure(extra={"epoch": epoch + 1})
            llm_logger.configure(extra={"epoch": epoch + 1})
            
            # Sample transitions
            self.sample_transitions()

            # Average reward of all sample in the buffer
            batched_transitions = BatchedTransitions.from_transitions(self._replay_buffer)
            batched_rewards = batched_transitions.reward
            avg_reward = np.mean(batched_rewards)
            hybrid_logger.info(f"Average sample reward: {avg_reward}")
            
            # Create a data loader that generates batched transitions
            replay_buffer_loader = ReplayBufferLoader(
                self._replay_buffer,
                batch_size=batch_size,
                min_batch_size=min_batch_size
            )
            
            # Train the agent
            self.update_rl_agent(
                optimizer=optimizer,
                replay_buffer_loader=replay_buffer_loader,
                n_steps_for_updating_agent=n_steps_for_updating_agent,
                ppo_epsilon=ppo_epsilon
            )
            
            # Clear replay buffer
            self._replay_buffer.clear()
            
    def sample_transitions(self) -> None:
        
        # Sample the first transition using RL agent
        self.sample_using_rl_agent()
        
        while len(self._replay_buffer) < self._replay_buffer.capacity:
            
            # Sample transitions using LLM agent
            if random.random() < self._llm_agent_sampling_freq:
                self.sample_using_llm_agent()
            
            # Sample transitions using RL agent
            else:
                self.sample_using_rl_agent()
            
    def sample_using_rl_agent(self) -> None:
        
        # Get current state
        state = self._state
        
        # Select an action
        action: Action = self._rl_agent.select_action(state)
        
        # Compute the log-probability of the action taken
        log_prob = self._rl_agent.log_prob(action)
        
        # Interact with the environment
        next_state, reward = self._env.step(action)
        
        # Log the performance result
        performance_result = PerformanceResult(
            hp_config=action.to_hp_config(),
            score=reward if reward is not None else 0.0
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
        
        # Add the transition to the replay buffer
        self._replay_buffer.add(transition)

        # Step to the next state
        self._state = next_state
        
    def sample_using_llm_agent(self) -> None:
        
        # Get current state
        state = self._state
        
        # Ask LLM to generate a new hyperparameter configuration setting
        action = self._llm_agent.select_action(self._performance_result_buffer)
        
        # If the action is None, then stop tuning
        if action is None:
            return
        
        # Interact with the environment
        next_state, reward = self._env.step(action)
        
        # Create a performance result
        performance_result = PerformanceResult(
            hp_config=action.to_hp_config(),
            score=reward if reward is not None else 0.0,
        )
        
        # Logging
        if reward is None:
            llm_logger.warning(f"{performance_result}; Time limit exceeded when fitting the model")
            
            # Set the reward to 0
            reward = 0.0
        else:
            llm_logger.info(performance_result)
            
        # Collect the performance result
        self._performance_result_buffer.push(performance_result)
        
        # Compute advantage
        advantage = self._advantage_calculator(reward)
        
        # Compute the log-probability
        log_prob = self._rl_agent.log_prob(action)
        
        # Collect the transition sample
        transition = Transition(
            state=state,
            action=action,
            reward=reward,
            advantage=advantage,
            log_prob=log_prob
        )
        self._replay_buffer.add(transition)
        
        # Step to the next state
        self._state = next_state

    def update_rl_agent(
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
                    curr_log_prob=self._rl_agent.log_prob(
                        batched_transitions.action,
                        batched_transitions.state
                    ),
                    old_log_prob=torch.tensor(batched_transitions.log_prob),
                    advantage=torch.tensor(batched_transitions.advantage),
                    epsilon=ppo_epsilon
                )
                
                # Update the agent
                
                # Clear the gradients
                optimizer.zero_grad()
                
                # Compute gradients
                loss.backward()
                
                # Clip the gradients
                torch.nn.utils.clip_grad_norm_(self._rl_agent.parameters(), 1.0)
                
                # Update the parameters of the network
                optimizer.step()
