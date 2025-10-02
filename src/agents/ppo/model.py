import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal
from typing import Tuple


class MLP(nn.Module):
    """Multi-layer perceptron backbone"""
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: Tuple[int, ...] = (64, 64),
        activation: nn.Module = nn.ReLU,
    ):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                activation(),
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, output_dim))
        self.network = nn.Sequential(*layers)
        
        # Orthogonal initialization
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                nn.init.constant_(module.bias, 0.0)
    
    def forward(self, x):
        return self.network(x)


class Actor(nn.Module):
    """Gaussian policy for continuous action space"""
    
    def __init__(
        self,
        input_dim: int,
        action_dim: int = 1,
        hidden_dims: Tuple[int, ...] = (64, 64),
        log_std_min: float = -20.0,
        log_std_max: float = 2.0,
    ):
        super().__init__()
        
        self.mean_net = MLP(
            input_dim=input_dim,
            output_dim=action_dim,
            hidden_dims=hidden_dims,
        )
        
        # Learnable log standard deviation (state-independent)
        self.log_std = nn.Parameter(torch.zeros(action_dim))
        
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.action_dim = action_dim
        
        # Small initialization for policy head
        nn.init.orthogonal_(self.mean_net.network[-1].weight, gain=0.01)
        nn.init.constant_(self.mean_net.network[-1].bias, 0.0)
    
    def forward(self, obs: torch.Tensor) -> Normal:
        """
        Return action distribution
        
        Args:
            obs: Flattened observation tensor (B, input_dim)
        
        Returns:
            Normal distribution over actions
        """
        mean = self.mean_net(obs)
        
        # Clamp log_std for numerical stability
        log_std = torch.clamp(self.log_std, self.log_std_min, self.log_std_max)
        std = torch.exp(log_std)
        
        return Normal(mean, std)
    
    def get_action(
        self, 
        obs: torch.Tensor, 
        deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample action and return log probability
        
        Args:
            obs: Flattened observation (B, input_dim)
            deterministic: If True, return mean action
        
        Returns:
            action: Sampled action (B, action_dim)
            log_prob: Log probability (B, 1)
        """
        dist = self.forward(obs)
        
        if deterministic:
            action = dist.mean
        else:
            action = dist.sample()
        
        log_prob = dist.log_prob(action).sum(dim=-1, keepdim=True)
        
        # Clip action to [0, 1] range (as per your action space)
        action = torch.clamp(action, 0.0, 1.0)
        
        return action, log_prob
    
    def evaluate_actions(
        self, 
        obs: torch.Tensor, 
        actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Evaluate log probability and entropy of given actions
        
        Args:
            obs: Flattened observations (B, input_dim)
            actions: Actions to evaluate (B, action_dim)
        
        Returns:
            log_prob: Log probabilities (B, 1)
            entropy: Entropy of distribution (B, 1)
        """
        dist = self.forward(obs)
        log_prob = dist.log_prob(actions).sum(dim=-1, keepdim=True)
        entropy = dist.entropy().sum(dim=-1, keepdim=True)
        
        return log_prob, entropy


class Critic(nn.Module):
    """State-value function approximator"""
    
    def __init__(
        self,
        input_dim: int,
        hidden_dims: Tuple[int, ...] = (64, 64),
    ):
        super().__init__()
        
        self.value_net = MLP(
            input_dim=input_dim,
            output_dim=1,
            hidden_dims=hidden_dims,
        )
    
    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Return state value estimate
        
        Args:
            obs: Flattened observation (B, input_dim)
        
        Returns:
            value: State value (B, 1)
        """
        return self.value_net(obs)

    

if __name__ == "__main__":
    pass