import gymnasium as gym
import envs

from envs.env_v0 import StoreConfig, Demand
from gymnasium.vector import SyncVectorEnv
from datetime import date
from agents.ppo.agent import PPOAgent

import numpy as np
import torch

# Environment configurations
store_config_1 = StoreConfig(seed=101, start_date=date(2021, 1, 1))
demand_1 = Demand(seed=1001, base=10.0)

store_config_2 = StoreConfig(seed=102, start_date=date(2021, 1, 1))
demand_2 = Demand(seed=1002, base=11.0)

store_config_3 = StoreConfig(seed=103, start_date=date(2021, 1, 1))
demand_3 = Demand(seed=1003, base=9.0)

store_config_4 = StoreConfig(seed=104, start_date=date(2021, 1, 1))
demand_4 = Demand(seed=1004, base=12.0)

store_config_5 = StoreConfig(seed=105, start_date=date(2021, 1, 1))
demand_5 = Demand(seed=1005, base=4.0)

env_id = "StoreEnv-v0"

env_fns = [
    lambda: gym.make(env_id, config=store_config_1, demand=demand_1), 
    lambda: gym.make(env_id, config=store_config_2, demand=demand_2), 
    lambda: gym.make(env_id, config=store_config_3, demand=demand_3), 
    lambda: gym.make(env_id, config=store_config_4, demand=demand_4), 
    lambda: gym.make(env_id, config=store_config_5, demand=demand_5), 
]


def run_vector_envs():
    """Main training function with episode-based learning"""
    
    # Create vectorized environment
    vector_env = SyncVectorEnv(env_fns)
    
    print(f"\n{'='*70}")
    print("Environment Setup")
    print(f"{'='*70}")
    print(f"Number of parallel environments: {vector_env.num_envs}")
    print(f"Observation space: {vector_env.observation_space}")
    print(f"Action space: {vector_env.action_space}")
    print(f"{'='*70}\n")
    
    # Create PPO agent
    agent = PPOAgent(
        vector_env=vector_env,
        output_dir="outputs",
        log_dir="logs",
        # Architecture
        hidden_dims=(64, 64),
        # Learning parameters
        learning_rate=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        vf_coef=0.5,
        ent_coef=0.01,
        max_grad_norm=0.5,
        # Training parameters (episode-based)
        n_episodes_per_update=10,   # Collect 10 episodes before each update
        n_epochs=10,                # 10 optimization epochs per update
        batch_size=64,
        total_episodes=1000,        # Train for 1000 episodes total
        # Evaluation
        eval_freq=50,               # Evaluate every 50 episodes
        n_eval_episodes=5,
        # Device
        device="auto",
    )
    
    # Train the agent
    print("Starting training...\n")
    agent.train()
    
    print("\nTraining completed! Running final evaluation...\n")
    
    # Final evaluation
    agent.eval(n_episodes=10)
    
    # Close environment
    vector_env.close()
    
    print("\nDone!")
    
    return agent


if __name__ == "__main__":
    # Set random seeds for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
        torch.backends.cudnn.deterministic = True
    
    print("="*70)
    print("PPO Training for Store Inventory Management (Episode-based)")
    print("="*70)
    
    # Run training
    trained_agent = run_vector_envs()