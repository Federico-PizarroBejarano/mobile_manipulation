"""Main training script for RL."""

import argparse
import time
from pathlib import Path

import numpy as np

from mm_rl.env.simple_goal_env import SimpleGoalEnv
from mm_rl.sac.sac import SAC
from mm_utils import parsing


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train RL agent with SAC")
    parser.add_argument(
        "-c", "--config", required=True, help="Path to configuration file"
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="./checkpoints",
        help="Directory to save checkpoints",
    )
    parser.add_argument(
        "--log_dir",
        type=str,
        default="./logs",
        help="Directory to save logs",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device to use (cpu or cuda)",
    )

    return parser.parse_args()


def evaluate(env, agent, n_episodes=5):
    """Evaluate agent performance.

    Args:
        env: Environment
        agent: SAC agent
        n_episodes: Number of episodes to evaluate

    Returns:
        dict: Evaluation metrics
    """
    episode_rewards = []
    episode_lengths = []
    success_count = 0

    for _ in range(n_episodes):
        obs, _ = env.reset()
        episode_reward = 0
        episode_length = 0
        done = False

        while not done:
            action = agent.select_action(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)

            episode_reward += reward
            episode_length += 1
            done = terminated or truncated

        # Check if goal was actually reached (not just early termination)
        ee_pos, ee_orn = env.sim.robot.link_pose()
        pos_error = np.linalg.norm(ee_pos - env.goal_pos)
        q_dot = np.abs(np.dot(ee_orn, env.goal_orn))
        orn_error = 1.0 - q_dot**2
        success = (
            terminated
            and pos_error <= env.success_pos_threshold
            and orn_error <= env.success_orn_threshold
        )
        if success:
            success_count += 1

        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)

    return {
        "mean_reward": np.mean(episode_rewards),
        "std_reward": np.std(episode_rewards),
        "mean_length": np.mean(episode_lengths),
        "success_rate": success_count / n_episodes,
    }


def train():
    """Main training loop."""
    args = parse_args()

    # Load configuration
    config = parsing.load_config(args.config)

    # Get RL config
    rl_config = config.get("rl")
    sac_config = rl_config.get("sac")

    # Create directories
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    # Initialize environment
    env = SimpleGoalEnv(config)

    # Get dimensions
    obs_shape = env.observation_space.shape
    action_shape = env.action_space.shape
    if obs_shape is None or action_shape is None:
        raise ValueError("Observation or action space shape is None")
    state_dim = obs_shape[0]
    action_dim = action_shape[0]
    # Action range for SAC: all actions are in [-1, 1] range
    # SAC will keep actions in this range, environment will unscale them
    action_range = (-1.0, 1.0)

    print(f"State dimension: {state_dim}")
    print(f"Action dimension: {action_dim}")
    print(f"Action range: {action_range}")

    # Initialize SAC agent
    agent = SAC(
        state_dim=state_dim,
        action_dim=action_dim,
        action_range=action_range,
        lr=sac_config.get("lr"),
        gamma=sac_config.get("gamma"),
        tau=sac_config.get("tau"),
        alpha=sac_config.get("alpha"),
        auto_alpha=sac_config.get("auto_alpha"),
        hidden_layers=sac_config["hidden_layers"],
        buffer_size=sac_config.get("buffer_size", 100000),
        device=args.device,
        infinite_horizon=sac_config.get("infinite_horizon", False),
    )

    # Training parameters
    max_steps = rl_config.get("max_steps")
    batch_size = sac_config.get("batch_size")
    start_steps = rl_config.get("start_steps")  # Random actions before training
    update_freq = rl_config.get("update_freq")  # Update every N steps
    update_after = rl_config.get("update_after")  # Start updating after N steps
    eval_freq = rl_config.get("eval_freq")  # Frequency of evaluation episodes
    save_freq = rl_config.get("save_freq")  # Frequency of checkpoint saving

    # Training loop
    obs, _ = env.reset()
    episode_reward = 0
    episode_length = 0
    total_steps = 0
    episode_count = 0

    # Metrics tracking
    episode_rewards = []
    training_metrics = []

    print("Starting training...")
    print(f"Max steps: {max_steps}")
    print(f"Start steps (random): {start_steps}")
    print(f"Update after: {update_after}")
    print(f"Eval frequency: {eval_freq}")
    print(f"Save frequency: {save_freq}")
    start_time = time.time()

    while total_steps < max_steps:
        # Select action
        if total_steps < start_steps:
            # Random action
            action = env.action_space.sample()
        else:
            # Policy action
            action = agent.select_action(obs)

        # Step environment
        next_obs, reward, terminated, truncated, _ = env.step(action)

        # Store transition
        done = terminated or truncated
        agent.replay_buffer.push(obs, action, reward, next_obs, done)

        obs = next_obs
        episode_reward += reward
        episode_length += 1
        total_steps += 1

        # Update agent
        if total_steps >= update_after and total_steps % update_freq == 0:
            metrics = agent.update(batch_size)
            if metrics:
                training_metrics.append({**metrics, "step": total_steps})

        # Episode done
        if done:
            episode_count += 1
            episode_rewards.append(episode_reward)

            # Log episode info
            if episode_count % 10 == 0:
                avg_reward = np.mean(episode_rewards[-10:])
                print(
                    f"Episode {episode_count}, Steps: {total_steps}, "
                    f"Reward: {episode_reward:.2f}, Avg (last 10): {avg_reward:.2f}, "
                    f"Length: {episode_length}"
                )

            # Reset environment
            obs, info = env.reset()
            episode_reward = 0
            episode_length = 0

        # Evaluation
        if total_steps % eval_freq == 0 and total_steps > 0:
            print(f"\nEvaluating at step {total_steps}...")
            eval_results = evaluate(env, agent, n_episodes=5)
            print(
                f"Eval - Mean reward: {eval_results['mean_reward']:.2f}, "
                f"Success rate: {eval_results['success_rate']:.2%}"
            )

        # Save checkpoint
        if total_steps % save_freq == 0 and total_steps > 0:
            checkpoint_path = checkpoint_dir / f"checkpoint_{total_steps}.pth"
            agent.save(str(checkpoint_path))
            print(f"Saved checkpoint to {checkpoint_path}")

    # Final evaluation
    print("\nFinal evaluation...")
    final_eval = evaluate(env, agent, n_episodes=10)
    print(
        f"Final Eval - Mean reward: {final_eval['mean_reward']:.2f}, "
        f"Success rate: {final_eval['success_rate']:.2%}"
    )

    # Save final model
    final_checkpoint = checkpoint_dir / "final_model.pth"
    agent.save(str(final_checkpoint))
    print(f"Saved final model to {final_checkpoint}")

    # Save training metrics
    if training_metrics:
        metrics_path = log_dir / "training_metrics.npz"
        # Convert to numpy arrays for saving
        metrics_dict = {}
        for key in training_metrics[0].keys():
            metrics_dict[key] = [m[key] for m in training_metrics]
        np.savez(str(metrics_path), **metrics_dict)
        print(f"Saved training metrics to {metrics_path}")

    env.close()

    elapsed_time = time.time() - start_time
    print(f"\nTraining completed in {elapsed_time:.2f} seconds")
    print(f"Total steps: {total_steps}, Episodes: {episode_count}")


if __name__ == "__main__":
    train()
