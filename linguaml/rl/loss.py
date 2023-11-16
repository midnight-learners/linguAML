import torch
from torch import Tensor

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
