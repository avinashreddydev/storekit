import numpy as np
import torch
from typing import Dict, Tuple
from dataclasses import dataclass, field

ORDER = ("inventory", "doy", "dow", "moy")


def obs_flatten(obs: Dict[str, np.ndarray]) -> np.ndarray:
    """Flatten dict observation to vector"""
    flat = np.concatenate([obs[k] for k in ORDER], axis=-1)  # (B, 4n)
    return flat


@dataclass
class RolloutBuffer:
    """Buffer for storing trajectory data during vectorized rollout"""
    
    observations: list = field(default_factory=list)  # List of flattened obs arrays
    actions: list = field(default_factory=list)
    rewards: list = field(default_factory=list)
    values: list = field(default_factory=list)
    log_probs: list = field(default_factory=list)
    dones: list = field(default_factory=list)
    
    def add(
        self,
        obs: np.ndarray,      # Already flattened (n_envs, obs_dim)
        action: np.ndarray,   # (n_envs, action_dim)
        reward: np.ndarray,   # (n_envs,)
        value: np.ndarray,    # (n_envs, 1)
        log_prob: np.ndarray, # (n_envs, 1)
        done: np.ndarray,     # (n_envs,)
    ):
        """Add a vectorized transition to the buffer"""
        self.observations.append(obs)
        self.actions.append(action)
        self.rewards.append(reward)
        self.values.append(value)
        self.log_probs.append(log_prob)
        self.dones.append(done)
    
    def get(self) -> Tuple[torch.Tensor, ...]:
        """Convert buffer to tensors"""
        obs = torch.FloatTensor(np.array(self.observations))          # (T, n_envs, obs_dim)
        actions = torch.FloatTensor(np.array(self.actions))           # (T, n_envs, action_dim)
        rewards = torch.FloatTensor(np.array(self.rewards))           # (T, n_envs)
        values = torch.FloatTensor(np.array(self.values)).squeeze(-1) # (T, n_envs)
        log_probs = torch.FloatTensor(np.array(self.log_probs))       # (T, n_envs, 1)
        dones = torch.FloatTensor(np.array(self.dones))               # (T, n_envs)
        
        return obs, actions, rewards, values, log_probs, dones
    
    def clear(self):
        """Clear the buffer"""
        self.observations.clear()
        self.actions.clear()
        self.rewards.clear()
        self.values.clear()
        self.log_probs.clear()
        self.dones.clear()
    
    def __len__(self):
        return len(self.rewards)


def compute_gae(
    rewards: torch.Tensor,      # (T, n_envs)
    values: torch.Tensor,       # (T, n_envs)
    dones: torch.Tensor,        # (T, n_envs)
    next_values: torch.Tensor,  # (n_envs,)
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute Generalized Advantage Estimation (GAE) for vectorized environments
    
    Returns:
        advantages: (T, n_envs)
        returns: (T, n_envs)
    """
    T, n_envs = rewards.shape
    advantages = torch.zeros_like(rewards)
    
    # Ensure last_gae is on the same device as input tensors
    last_gae = torch.zeros(n_envs, device=rewards.device)
    
    # Backward pass to compute advantages
    for t in reversed(range(T)):
        if t == T - 1:
            next_val = next_values
        else:
            next_val = values[t + 1]
        
        # TD error: δ_t = r_t + γ*V(s_{t+1}) - V(s_t)
        delta = rewards[t] + gamma * next_val * (1.0 - dones[t]) - values[t]
        
        # GAE: A_t = δ_t + γ*λ*A_{t+1}
        last_gae = delta + gamma * gae_lambda * (1.0 - dones[t]) * last_gae
        advantages[t] = last_gae
    
    # Returns are advantages + values
    returns = advantages + values
    
    return advantages, returns


def normalize_advantages(advantages: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Normalize advantages to have zero mean and unit variance"""
    return (advantages - advantages.mean()) / (advantages.std() + eps)

if __name__ == "__main__":
    pass