from agents.ppo.model import Actor, Critic
from agents.ppo.utils import (
    obs_flatten, 
    ORDER, 
    RolloutBuffer, 
    compute_gae, 
    normalize_advantages
)
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, Tuple
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
from loguru import logger


class PPOAgent:
    """PPO Agent for vectorized environments - Episode-based training"""
    
    def __init__(
        self, 
        vector_env, 
        output_dir: str = "outputs", 
        log_dir: str = "logs",
        # Architecture
        hidden_dims: tuple = (64, 64),
        # Learning
        learning_rate: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_range: float = 0.2,
        vf_coef: float = 0.5,
        ent_coef: float = 0.01,
        max_grad_norm: float = 0.5,
        # Training
        n_episodes_per_update: int = 10,  # Episodes to collect before update
        n_epochs: int = 10,               # Update epochs
        batch_size: int = 64,             # Mini-batch size
        total_episodes: int = 1000,       # Total training episodes
        # Evaluation
        eval_freq: int = 50,              # Evaluate every N episodes
        n_eval_episodes: int = 5,
        # Device
        device: str = "auto",
    ):
        self.env = vector_env
        self.n_envs = vector_env.num_envs
        
        # Get observation and action dimensions
        sample_obs = vector_env.observation_space.sample()
        self.obs_dim = obs_flatten(sample_obs).shape[-1]  # 4 * hist_length
        self.action_dim = vector_env.action_space.shape[-1]
        
        # Hyperparameters
        self.n_episodes_per_update = n_episodes_per_update
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.total_episodes = total_episodes
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_range = clip_range
        self.vf_coef = vf_coef
        self.ent_coef = ent_coef
        self.max_grad_norm = max_grad_norm
        
        # Evaluation
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        
        # Device
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        logger.info(f"Using device: {self.device}")
        logger.info(f"Observation dim: {self.obs_dim}, Action dim: {self.action_dim}")
        logger.info(f"Number of parallel environments: {self.n_envs}")
        
        # Initialize networks
        self.actor = Actor(
            input_dim=self.obs_dim,
            action_dim=self.action_dim,
            hidden_dims=hidden_dims,
        ).to(self.device)
        
        self.critic = Critic(
            input_dim=self.obs_dim,
            hidden_dims=hidden_dims,
        ).to(self.device)
        
        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=learning_rate)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=learning_rate)
        
        # Rollout buffer
        self.buffer = RolloutBuffer()
        
        # Directories
        self.output_dir = Path(output_dir)
        self.log_dir = Path(log_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Configure loguru
        logger.add(
            self.log_dir / "training.log",
            rotation="10 MB",
            level="INFO",
            format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}"
        )
        
        # TensorBoard
        self.writer = SummaryWriter(log_dir=str(self.log_dir / "tensorboard"))
        
        # Training state
        self.num_episodes = 0
        self.num_timesteps = 0
        self.num_updates = 0
        
        # Tracking
        self.episode_rewards = []
        self.episode_lengths = []
        self.recent_rewards = []  # Last 100 episodes
    
    def select_action(
        self, 
        obs_dict: Dict[str, np.ndarray],
        deterministic: bool = False
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Select actions for all environments
        
        Returns:
            actions: (n_envs, action_dim)
            values: (n_envs, 1)
            log_probs: (n_envs, 1)
        """
        # Flatten observations
        obs_flat = obs_flatten(obs_dict)  # (n_envs, obs_dim)
        obs_torch = torch.FloatTensor(obs_flat).to(self.device)
        
        with torch.no_grad():
            # Get action from actor
            actions, log_probs = self.actor.get_action(obs_torch, deterministic)
            
            # Get value from critic
            values = self.critic(obs_torch)
        
        return (
            actions.cpu().numpy(),
            values.cpu().numpy(),
            log_probs.cpu().numpy(),
        )
    
    def collect_episodes(self, n_episodes: int) -> Tuple[Dict[str, float], np.ndarray]:
        """
        Collect n_episodes of experience from vectorized environments
        
        Returns:
            stats: Dictionary with rollout statistics
            next_values: Bootstrap values for last states
        """
        self.buffer.clear()
        
        obs_dict, _ = self.env.reset()
        
        # Track per-env episode stats
        current_ep_rewards = np.zeros(self.n_envs)
        current_ep_lengths = np.zeros(self.n_envs)
        episodes_collected = 0
        
        episode_rewards_list = []
        episode_lengths_list = []
        
        while episodes_collected < n_episodes:
            # Select actions for all envs
            actions, values, log_probs = self.select_action(obs_dict, deterministic=False)
            
            # Step all environments
            next_obs_dict, rewards, terminated, truncated, infos = self.env.step(actions)
            
            # Combine terminated and truncated
            dones = np.logical_or(terminated, truncated)
            
            # Flatten current observation for storage
            obs_flat = obs_flatten(obs_dict)
            
            # Store transitions
            self.buffer.add(
                obs=obs_flat,
                action=actions,
                reward=rewards,
                value=values,
                log_prob=log_probs,
                done=dones,
            )
            
            # Update episode stats
            current_ep_rewards += rewards
            current_ep_lengths += 1
            self.num_timesteps += self.n_envs
            
            # Handle episode ends
            for i, done in enumerate(dones):
                if done:
                    episode_rewards_list.append(current_ep_rewards[i])
                    episode_lengths_list.append(current_ep_lengths[i])
                    
                    # Log individual episode
                    logger.debug(
                        f"Env {i} | Episode {self.num_episodes + episodes_collected + 1} | "
                        f"Reward: {current_ep_rewards[i]:.2f} | "
                        f"Length: {int(current_ep_lengths[i])}"
                    )
                    
                    current_ep_rewards[i] = 0.0
                    current_ep_lengths[i] = 0
                    episodes_collected += 1
                    
                    if episodes_collected >= n_episodes:
                        break
            
            obs_dict = next_obs_dict
            
            if episodes_collected >= n_episodes:
                break
        
        # Compute value of last state for GAE (bootstrap)
        obs_flat = obs_flatten(obs_dict)
        obs_torch = torch.FloatTensor(obs_flat).to(self.device)
        
        with torch.no_grad():
            next_values = self.critic(obs_torch).cpu().numpy().squeeze()
        
        # Update global tracking
        self.episode_rewards.extend(episode_rewards_list)
        self.episode_lengths.extend(episode_lengths_list)
        self.recent_rewards.extend(episode_rewards_list)
        self.recent_rewards = self.recent_rewards[-100:]  # Keep last 100
        
        stats = {
            "episodes_collected": len(episode_rewards_list),
            "ep_rew_mean": np.mean(episode_rewards_list) if episode_rewards_list else 0.0,
            "ep_rew_std": np.std(episode_rewards_list) if episode_rewards_list else 0.0,
            "ep_len_mean": np.mean(episode_lengths_list) if episode_lengths_list else 0.0,
            "ep_rew_recent_100": np.mean(self.recent_rewards) if self.recent_rewards else 0.0,
        }
        
        return stats, next_values
    
    def update(self, next_values: np.ndarray) -> Dict[str, float]:
        """
        Update policy and value function using PPO
        
        Args:
            next_values: Bootstrap values (n_envs,)
        
        Returns:
            Dictionary with training statistics
        """
        # Get data from buffer
        obs, actions, rewards, old_values, old_log_probs, dones = self.buffer.get()
        
        # Move to device
        obs = obs.to(self.device)
        actions = actions.to(self.device)
        old_log_probs = old_log_probs.to(self.device)
        
        # Compute advantages and returns
        next_values_tensor = torch.FloatTensor(next_values).to(self.device)
        advantages, returns = compute_gae(
            rewards=rewards.to(self.device),
            values=old_values.to(self.device),
            dones=dones.to(self.device),
            next_values=next_values_tensor,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
        )
        
        # Flatten batch dimensions: (T, n_envs, ...) -> (T*n_envs, ...)
        obs = obs.view(-1, obs.shape[-1])
        actions = actions.view(-1, actions.shape[-1])
        old_log_probs = old_log_probs.view(-1, 1)
        advantages = advantages.view(-1)
        returns = returns.view(-1)
        
        # Normalize advantages
        advantages = normalize_advantages(advantages)
        
        # Training metrics
        policy_losses = []
        value_losses = []
        entropy_losses = []
        clip_fractions = []
        
        # Multiple epochs of optimization
        for epoch in range(self.n_epochs):
            # Shuffle data
            batch_size = obs.shape[0]
            indices = torch.randperm(batch_size)
            
            for start in range(0, batch_size, self.batch_size):
                end = min(start + self.batch_size, batch_size)
                batch_indices = indices[start:end]
                
                # Get mini-batch
                obs_batch = obs[batch_indices]
                actions_batch = actions[batch_indices]
                old_log_probs_batch = old_log_probs[batch_indices]
                advantages_batch = advantages[batch_indices].unsqueeze(1)
                returns_batch = returns[batch_indices]
                
                # Evaluate actions with current policy
                new_log_probs, entropy = self.actor.evaluate_actions(obs_batch, actions_batch)
                new_values = self.critic(obs_batch).squeeze()
                
                # Policy loss (PPO clip objective)
                ratio = torch.exp(new_log_probs - old_log_probs_batch)
                surr1 = ratio * advantages_batch
                surr2 = torch.clamp(
                    ratio, 
                    1.0 - self.clip_range, 
                    1.0 + self.clip_range
                ) * advantages_batch
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Value loss (MSE)
                value_loss = nn.functional.mse_loss(new_values, returns_batch)
                
                # Entropy bonus (encourage exploration)
                entropy_loss = -entropy.mean()
                
                # Total loss
                loss = (
                    policy_loss 
                    + self.vf_coef * value_loss 
                    + self.ent_coef * entropy_loss
                )
                
                # Optimize
                self.actor_optimizer.zero_grad()
                self.critic_optimizer.zero_grad()
                loss.backward()
                
                # Clip gradients
                nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                
                self.actor_optimizer.step()
                self.critic_optimizer.step()
                
                # Collect metrics
                policy_losses.append(policy_loss.item())
                value_losses.append(value_loss.item())
                entropy_losses.append(entropy_loss.item())
                
                with torch.no_grad():
                    clip_fraction = ((ratio - 1.0).abs() > self.clip_range).float().mean()
                    clip_fractions.append(clip_fraction.item())
        
        self.num_updates += 1
        
        stats = {
            "policy_loss": np.mean(policy_losses),
            "value_loss": np.mean(value_losses),
            "entropy_loss": np.mean(entropy_losses),
            "clip_fraction": np.mean(clip_fractions),
        }
        
        return stats
    
    def evaluate(self) -> Dict[str, float]:
        """Evaluate current policy"""
        eval_rewards = []
        eval_lengths = []
        
        for ep in range(self.n_eval_episodes):
            obs_dict, _ = self.env.reset()
            done = False
            ep_reward = 0.0
            ep_length = 0
            
            while not done:
                actions, _, _ = self.select_action(obs_dict, deterministic=True)
                obs_dict, rewards, terminated, truncated, _ = self.env.step(actions)
                
                # Take first env for evaluation
                ep_reward += rewards[0]
                ep_length += 1
                done = terminated[0] or truncated[0]
            
            eval_rewards.append(ep_reward)
            eval_lengths.append(ep_length)
        
        stats = {
            "eval_rew_mean": np.mean(eval_rewards),
            "eval_rew_std": np.std(eval_rewards),
            "eval_len_mean": np.mean(eval_lengths),
        }
        
        return stats
    
    def train(self):
        """Main training loop - episode-based"""
        logger.info("=" * 60)
        logger.info("Starting PPO Training (Episode-based)")
        logger.info("=" * 60)
        logger.info(f"Total episodes: {self.total_episodes:,}")
        logger.info(f"Episodes per update: {self.n_episodes_per_update}")
        logger.info(f"Parallel environments: {self.n_envs}")
        logger.info(f"Update epochs: {self.n_epochs}")
        logger.info(f"Batch size: {self.batch_size}")
        logger.info("=" * 60)
        
        while self.num_episodes < self.total_episodes:
            # Collect episodes
            logger.info(f"\nCollecting {self.n_episodes_per_update} episodes...")
            rollout_stats, next_values = self.collect_episodes(self.n_episodes_per_update)
            
            self.num_episodes += rollout_stats["episodes_collected"]
            
            # Update policy
            logger.info("Updating policy...")
            train_stats = self.update(next_values)
            
            # Log to TensorBoard
            self.writer.add_scalar("rollout/ep_rew_mean", rollout_stats["ep_rew_mean"], self.num_episodes)
            self.writer.add_scalar("rollout/ep_rew_std", rollout_stats["ep_rew_std"], self.num_episodes)
            self.writer.add_scalar("rollout/ep_len_mean", rollout_stats["ep_len_mean"], self.num_episodes)
            self.writer.add_scalar("rollout/ep_rew_recent_100", rollout_stats["ep_rew_recent_100"], self.num_episodes)
            
            self.writer.add_scalar("train/policy_loss", train_stats["policy_loss"], self.num_episodes)
            self.writer.add_scalar("train/value_loss", train_stats["value_loss"], self.num_episodes)
            self.writer.add_scalar("train/entropy_loss", train_stats["entropy_loss"], self.num_episodes)
            self.writer.add_scalar("train/clip_fraction", train_stats["clip_fraction"], self.num_episodes)
            
            self.writer.add_scalar("time/num_updates", self.num_updates, self.num_episodes)
            self.writer.add_scalar("time/num_timesteps", self.num_timesteps, self.num_episodes)
            
            # Log progress
            logger.info(
                f"Episodes: {self.num_episodes}/{self.total_episodes} | "
                f"Avg Reward: {rollout_stats['ep_rew_mean']:.2f} ± {rollout_stats['ep_rew_std']:.2f} | "
                f"Avg Length: {rollout_stats['ep_len_mean']:.1f} | "
                f"Recent 100: {rollout_stats['ep_rew_recent_100']:.2f}"
            )
            logger.info(
                f"Training: Policy Loss={train_stats['policy_loss']:.4f} | "
                f"Value Loss={train_stats['value_loss']:.4f} | "
                f"Entropy={train_stats['entropy_loss']:.4f} | "
                f"Clip Frac={train_stats['clip_fraction']:.3f}"
            )
            
            # Periodic evaluation
            if self.num_episodes % self.eval_freq < self.n_episodes_per_update:
                logger.info("Running evaluation...")
                eval_stats = self.evaluate()
                
                self.writer.add_scalar("eval/ep_rew_mean", eval_stats["eval_rew_mean"], self.num_episodes)
                self.writer.add_scalar("eval/ep_rew_std", eval_stats["eval_rew_std"], self.num_episodes)
                self.writer.add_scalar("eval/ep_len_mean", eval_stats["eval_len_mean"], self.num_episodes)
                
                logger.info(
                    f"[EVAL] Reward: {eval_stats['eval_rew_mean']:.2f} ± {eval_stats['eval_rew_std']:.2f} | "
                    f"Length: {eval_stats['eval_len_mean']:.1f}"
                )
                
                # Save checkpoint
                checkpoint_path = self.output_dir / f"checkpoint_ep{self.num_episodes}.pt"
                self.save(checkpoint_path)
        
        # Final save
        final_path = self.output_dir / "final_model.pt"
        self.save(final_path)
        self.writer.close()
        
        logger.info("=" * 60)
        logger.info("Training Complete!")
        logger.info(f"Total episodes: {self.num_episodes}")
        logger.info(f"Total timesteps: {self.num_timesteps:,}")
        logger.info(f"Final avg reward (last 100): {np.mean(self.recent_rewards):.2f}")
        logger.info("=" * 60)
    
    def eval(self, n_episodes: int = 10):
        """Run evaluation episodes"""
        logger.info(f"\nRunning {n_episodes} evaluation episodes...")
        
        eval_rewards = []
        eval_lengths = []
        
        for ep in range(n_episodes):
            obs_dict, _ = self.env.reset()
            done = False
            ep_reward = 0.0
            ep_length = 0
            
            while not done:
                actions, _, _ = self.select_action(obs_dict, deterministic=True)
                obs_dict, rewards, terminated, truncated, infos = self.env.step(actions)
                
                ep_reward += rewards[0]
                ep_length += 1
                done = terminated[0] or truncated[0]
            
            eval_rewards.append(ep_reward)
            eval_lengths.append(ep_length)
            
            logger.info(f"Episode {ep+1}/{n_episodes}: Reward={ep_reward:.2f}, Length={ep_length}")
        
        logger.info("\nEvaluation Results:")
        logger.info(f"  Mean Reward: {np.mean(eval_rewards):.2f} ± {np.std(eval_rewards):.2f}")
        logger.info(f"  Mean Length: {np.mean(eval_lengths):.1f} ± {np.std(eval_lengths):.1f}")
    
    def save(self, path):
        """Save model checkpoint"""
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
            'num_episodes': self.num_episodes,
            'num_timesteps': self.num_timesteps,
            'num_updates': self.num_updates,
            'episode_rewards': self.episode_rewards,
            'recent_rewards': self.recent_rewards,
        }, path)
        logger.info(f"Model saved to {path}")
    
    def load(self, path):
        """Load model checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
        self.num_episodes = checkpoint['num_episodes']
        self.num_timesteps = checkpoint['num_timesteps']
        self.num_updates = checkpoint['num_updates']
        self.episode_rewards = checkpoint.get('episode_rewards', [])
        self.recent_rewards = checkpoint.get('recent_rewards', [])
        logger.info(f"Model loaded from {path}")