# %%
import gymnasium as gym
import envs

from envs.env_v0 import StoreConfig, Demand
from gymnasium.vector import SyncVectorEnv
from datetime import date
from agents.ppo.agent import PPOAgent
from agents.constant.agent import ConstantAgent

import numpy as np
import torch
import os
from loguru import logger

# ============================================================================
# Environment Configurations
# ============================================================================

# Training environments
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

# Evaluation environment (1 year only)
eval_config_1 = StoreConfig(seed=106, start_date=date(2021, 1, 1), end_date=date(2021, 12, 31))
eval_demand_1 = Demand(seed=1006, base=10.0)

env_id = "StoreEnv-v0"

# Training environment factory functions
env_fns = [
    lambda: gym.make(env_id, config=store_config_1, demand=demand_1), 
    lambda: gym.make(env_id, config=store_config_2, demand=demand_2), 
    lambda: gym.make(env_id, config=store_config_3, demand=demand_3), 
    lambda: gym.make(env_id, config=store_config_4, demand=demand_4), 
    lambda: gym.make(env_id, config=store_config_5, demand=demand_5), 
]


# ============================================================================
# Training Functions
# ============================================================================

def run_vector_envs():
    """Main training function with episode-based learning"""
    
    # Create vectorized environment
    vector_env = SyncVectorEnv(env_fns)
    
    # Create evaluation environment
    eval_env = gym.make(env_id, config=eval_config_1, demand=eval_demand_1)
    
    logger.info("=" * 70)
    logger.info("Environment Setup")
    logger.info("=" * 70)
    logger.info(f"Number of parallel environments: {vector_env.num_envs}")
    logger.info(f"Training observation space: {vector_env.observation_space}")
    logger.info(f"Training action space: {vector_env.action_space}")
    logger.info(f"Evaluation observation space: {eval_env.observation_space}")
    logger.info(f"Evaluation action space: {eval_env.action_space}")
    logger.info("=" * 70)
    
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
    logger.info("Starting PPO training...")
    agent.train()
    
    logger.info("Training completed! Running final evaluation...")
    
    # Final evaluation on eval environment
    agent.eval(n_episodes=10)
    
    # Close environments
    vector_env.close()
    eval_env.close()
    
    logger.info("Done!")
    
    return agent


def run_constant_agent(constant: float = 0.3, n_episodes: int = 10):
    """
    Evaluate a constant action baseline agent
    
    Args:
        constant: Constant action value (0-1)
        n_episodes: Number of episodes to evaluate
    """
    logger.info("=" * 70)
    logger.info("Running Constant Agent Baseline")
    logger.info("=" * 70)
    logger.info(f"Constant action: {constant}")
    logger.info(f"Number of episodes: {n_episodes}")
    logger.info("=" * 70)
    
    # Create evaluation environment
    eval_env = gym.make(env_id, config=eval_config_1, demand=eval_demand_1)
    
    # Create constant agent
    agent = ConstantAgent(eval_env, constant=constant)
    
    # Evaluate
    stats = agent.eval(n_episodes=n_episodes)
    
    # Plot results
    agent.plot(save_dir="plots")
    
    # Close environment
    eval_env.close()
    
    logger.info("Constant agent evaluation complete!")
    
    return agent


def compare_constant_agents(constants: list = None, n_episodes: int = 10):
    """
    Compare multiple constant action values
    
    Args:
        constants: List of constant values to test
        n_episodes: Number of episodes per constant
    """
    if constants is None:
        constants = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    
    logger.info("=" * 70)
    logger.info("Comparing Constant Agents")
    logger.info("=" * 70)
    logger.info(f"Testing constants: {constants}")
    logger.info(f"Episodes per constant: {n_episodes}")
    logger.info("=" * 70)
    
    # Create evaluation environment
    eval_env = gym.make(env_id, config=eval_config_1, demand=eval_demand_1)
    
    # Compare
    results = ConstantAgent.compare_constants(
        eval_env, 
        constants=constants, 
        n_episodes=n_episodes,
        save_dir="plots"
    )
    
    # Close environment
    eval_env.close()
    
    logger.info("Constant agent comparison complete!")
    
    return results


# ============================================================================
# Main Entry Point
# ============================================================================

if __name__ == "__main__":
    # Set random seeds for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
        torch.backends.cudnn.deterministic = True
    
    # Create necessary directories
    os.makedirs("plots", exist_ok=True)
    os.makedirs("outputs", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    
    logger.info("=" * 70)
    logger.info("Store Inventory Management - RL Training")
    logger.info("=" * 70)
    
    # Choose what to run
    mode = "compare"  # Options: "constant", "compare", "train"
    
    if mode == "constant":
        # Run single constant agent
        logger.info("Mode: Single Constant Agent Evaluation")
        agent = run_constant_agent(constant=0.3, n_episodes=10)
        
    elif mode == "compare":
        # Compare multiple constant values
        logger.info("Mode: Compare Multiple Constant Actions")
        results = compare_constant_agents(
            constants=[ 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            n_episodes=10
        )
        
    elif mode == "train":
        # Train PPO agent
        logger.info("Mode: PPO Training")
        trained_agent = run_vector_envs()
        
    else:
        logger.error(f"Unknown mode: {mode}")
        logger.info("Available modes: 'constant', 'compare', 'train'")
    
    logger.info("=" * 70)
    logger.info("All tasks completed!")
    logger.info("=" * 70)