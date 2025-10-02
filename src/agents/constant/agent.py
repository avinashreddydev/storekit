import numpy as np
import matplotlib.pyplot as plt
from loguru import logger
from tqdm import tqdm
from pathlib import Path


class ConstantAgent:
    """Baseline agent that always takes a constant action"""
    
    def __init__(self, eval_env, constant: float):
        """
        Args:
            eval_env: Single (non-vectorized) gym environment
            constant: Constant action value to take (between 0 and 1)
        """
        self.constant = np.clip(constant, 0.0, 1.0)
        self.eval_env = eval_env
        
        # Storage for metrics
        self.episode_rewards = []
        self.episode_lengths = []
        self.unmet_demands = []
        self.holding_costs = []
        
        logger.info(f"ConstantAgent initialized with action={self.constant:.2f}")
    
    def eval(self, n_episodes: int = 10) -> dict:
        """
        Evaluate the constant agent over multiple episodes
        
        Args:
            n_episodes: Number of episodes to run
            
        Returns:
            Dictionary with evaluation statistics
        """
        logger.info(f"Evaluating ConstantAgent for {n_episodes} episodes...")
        
        for ep in tqdm(range(n_episodes), desc="Evaluating"):
            obs, info = self.eval_env.reset()
            
            episode_reward = 0.0
            episode_length = 0
            episode_unmet = []
            episode_holding = []
            
            terminated = False
            truncated = False
            
            while not (terminated or truncated):
                # Take constant action
                action = np.array([self.constant], dtype=np.float32)
                obs, reward, terminated, truncated, info = self.eval_env.step(action)
                
                # Accumulate metrics
                episode_reward += reward
                episode_length += 1
                episode_unmet.append(info["unmet_demand"])
                episode_holding.append(info["holding_cost"])
            
            # Store episode results
            self.episode_rewards.append(episode_reward)
            self.episode_lengths.append(episode_length)
            self.unmet_demands.append(episode_unmet)
            self.holding_costs.append(episode_holding)
            
            logger.debug(
                f"Episode {ep+1}/{n_episodes}: "
                f"Reward={episode_reward:.2f}, "
                f"Length={episode_length}, "
                f"Total Unmet={sum(episode_unmet):.2f}, "
                f"Total Holding={sum(episode_holding):.2f}"
            )
        
        # Calculate statistics
        stats = {
            "mean_reward": np.mean(self.episode_rewards),
            "std_reward": np.std(self.episode_rewards),
            "mean_length": np.mean(self.episode_lengths),
            "mean_unmet_demand": np.mean([sum(ep) for ep in self.unmet_demands]),
            "mean_holding_cost": np.mean([sum(ep) for ep in self.holding_costs]),
            "total_cost": np.mean([sum(ep_u) + sum(ep_h) 
                                  for ep_u, ep_h in zip(self.unmet_demands, self.holding_costs)])
        }
        
        logger.info("\nEvaluation Results:")
        logger.info(f"  Mean Reward: {stats['mean_reward']:.2f} ± {stats['std_reward']:.2f}")
        logger.info(f"  Mean Length: {stats['mean_length']:.1f}")
        logger.info(f"  Mean Unmet Demand: {stats['mean_unmet_demand']:.2f}")
        logger.info(f"  Mean Holding Cost: {stats['mean_holding_cost']:.2f}")
        logger.info(f"  Mean Total Cost: {stats['total_cost']:.2f}")
        
        return stats
    
    def plot(self, save_dir: str = "plots"):
        """
        Plot evaluation results
        
        Args:
            save_dir: Directory to save plots
        """
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        
        if not self.episode_rewards:
            logger.warning("No evaluation data to plot. Run eval() first.")
            return
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f'Constant Agent Evaluation (action={self.constant:.2f})', 
                     fontsize=14, fontweight='bold')
        
        # Plot 1: Episode Rewards
        axes[0, 0].plot(self.episode_rewards, marker='o', linestyle='-', alpha=0.7)
        axes[0, 0].axhline(y=np.mean(self.episode_rewards), color='r', 
                          linestyle='--', label=f'Mean: {np.mean(self.episode_rewards):.2f}')
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Total Reward')
        axes[0, 0].set_title('Episode Rewards')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Costs over time (first episode)
        if self.unmet_demands:
            steps = range(len(self.unmet_demands[0]))
            axes[0, 1].plot(steps, self.unmet_demands[0], label='Unmet Demand', alpha=0.7)
            axes[0, 1].plot(steps, self.holding_costs[0], label='Holding Cost', alpha=0.7)
            axes[0, 1].set_xlabel('Step')
            axes[0, 1].set_ylabel('Cost')
            axes[0, 1].set_title('Costs Over Time (Episode 1)')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Total costs per episode
        total_unmet = [sum(ep) for ep in self.unmet_demands]
        total_holding = [sum(ep) for ep in self.holding_costs]
        
        x = np.arange(len(total_unmet))
        width = 0.35
        
        axes[1, 0].bar(x - width/2, total_unmet, width, label='Unmet Demand', alpha=0.7)
        axes[1, 0].bar(x + width/2, total_holding, width, label='Holding Cost', alpha=0.7)
        axes[1, 0].set_xlabel('Episode')
        axes[1, 0].set_ylabel('Total Cost')
        axes[1, 0].set_title('Total Costs per Episode')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3, axis='y')
        
        # Plot 4: Cost distribution (box plot)
        axes[1, 1].boxplot([total_unmet, total_holding], 
                          labels=['Unmet Demand', 'Holding Cost'])
        axes[1, 1].set_ylabel('Cost')
        axes[1, 1].set_title('Cost Distribution Across Episodes')
        axes[1, 1].grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        # Save figure
        plot_path = save_path / f'constant_agent_{self.constant:.2f}_evaluation.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        logger.info(f"Plot saved to {plot_path}")
        
        plt.close()
    
    def compare_constants(eval_env, constants: list, n_episodes: int = 10, save_dir: str = "plots"):
        """
        Compare multiple constant action values
        
        Args:
            eval_env: Environment for evaluation
            constants: List of constant action values to test
            n_episodes: Number of episodes per constant
            save_dir: Directory to save comparison plot
        """
        logger.info(f"Comparing {len(constants)} constant actions...")
        
        results = []
        
        for const in constants:
            agent = ConstantAgent(eval_env, const)
            stats = agent.eval(n_episodes)
            results.append({
                'constant': const,
                'mean_reward': stats['mean_reward'],
                'std_reward': stats['std_reward'],
                'total_cost': stats['total_cost'],
                'unmet_demand': stats['mean_unmet_demand'],
                'holding_cost': stats['mean_holding_cost']
            })
        
        # Plot comparison
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        fig.suptitle('Constant Agent Comparison', fontsize=14, fontweight='bold')
        
        constants_list = [r['constant'] for r in results]
        rewards = [r['mean_reward'] for r in results]
        reward_stds = [r['std_reward'] for r in results]
        unmets = [r['unmet_demand'] for r in results]
        holdings = [r['holding_cost'] for r in results]
        
        # Plot 1: Mean Rewards
        axes[0].errorbar(constants_list, rewards, yerr=reward_stds, 
                        marker='o', capsize=5, capthick=2)
        axes[0].set_xlabel('Constant Action')
        axes[0].set_ylabel('Mean Reward')
        axes[0].set_title('Mean Reward vs Action')
        axes[0].grid(True, alpha=0.3)
        
        # Plot 2: Cost breakdown
        x = np.arange(len(constants_list))
        width = 0.35
        axes[1].bar(x - width/2, unmets, width, label='Unmet Demand', alpha=0.7)
        axes[1].bar(x + width/2, holdings, width, label='Holding Cost', alpha=0.7)
        axes[1].set_xlabel('Constant Action')
        axes[1].set_ylabel('Mean Cost')
        axes[1].set_title('Cost Breakdown')
        axes[1].set_xticks(x)
        axes[1].set_xticklabels([f'{c:.2f}' for c in constants_list])
        axes[1].legend()
        axes[1].grid(True, alpha=0.3, axis='y')
        
        # Plot 3: Total cost
        total_costs = [r['total_cost'] for r in results]
        axes[2].plot(constants_list, total_costs, marker='o', linestyle='-', linewidth=2)
        axes[2].set_xlabel('Constant Action')
        axes[2].set_ylabel('Total Cost')
        axes[2].set_title('Total Cost vs Action')
        axes[2].grid(True, alpha=0.3)
        
        # Mark best action
        best_idx = np.argmax(rewards)
        axes[2].axvline(x=constants_list[best_idx], color='r', 
                       linestyle='--', alpha=0.7, 
                       label=f'Best: {constants_list[best_idx]:.2f}')
        axes[2].legend()
        
        plt.tight_layout()
        
        plot_path = save_path / 'constant_agent_comparison.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        logger.info(f"Comparison plot saved to {plot_path}")
        
        plt.close()
        
        # Print best result
        best_result = results[best_idx]
        logger.info(f"\nBest constant action: {best_result['constant']:.2f}")
        logger.info(f"  Mean Reward: {best_result['mean_reward']:.2f} ± {best_result['std_reward']:.2f}")
        logger.info(f"  Total Cost: {best_result['total_cost']:.2f}")
        
        return results


# Example usage
if __name__ == "__main__":
    pass