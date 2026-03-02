"""Evaluate a trained model with visualization.

Command:

```bash
cd /home/federico/catkin_ws/src/mobile_manipulation
python3 mm_rl/evaluate_model.py -c mm_rl/config/train_config.yaml --checkpoint checkpoints/final_model.pth --gui --n-episodes 3
"""

import argparse
import time
from pathlib import Path

import numpy as np

from mm_rl.env.simple_goal_env import SimpleGoalEnv
from mm_rl.sac.sac import SAC
from mm_utils import parsing


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate trained RL model")
    parser.add_argument(
        "-c", "--config", required=True, help="Path to configuration file"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint (.pth file)",
    )
    parser.add_argument(
        "--n-episodes",
        type=int,
        default=5,
        help="Number of episodes to run",
    )
    parser.add_argument(
        "--gui",
        action="store_true",
        help="Enable GUI visualization",
    )
    parser.add_argument(
        "--save-trajectory",
        action="store_true",
        help="Save trajectory data to file",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./eval_results",
        help="Directory to save evaluation results",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for evaluation",
    )

    return parser.parse_args()


def evaluate_episode(env, agent, episode_num, save_trajectory=False):
    """Run a single evaluation episode.

    Args:
        env: Environment
        agent: SAC agent
        episode_num: Episode number
        save_trajectory: Whether to save trajectory data

    Returns:
        dict: Episode results
    """
    obs, info = env.reset()
    episode_reward = 0
    episode_length = 0
    done = False

    # Store trajectory if requested
    trajectory = (
        {
            "observations": [],
            "actions": [],
            "rewards": [],
            "ee_positions": [],
            "ee_orientations": [],
            "goal_positions": [],
            "goal_orientations": [],
            "base_positions": [],
        }
        if save_trajectory
        else None
    )

    print(f"\nEpisode {episode_num}:")
    print(f"  Goal position: {info['goal_pos']}")
    print(f"  Goal orientation: {info['goal_orn']}")

    start_time = time.time()

    while not done:
        # Select action (deterministic for evaluation)
        action = agent.select_action(obs, deterministic=True)

        # Step environment
        obs, reward, terminated, truncated, info = env.step(action)

        episode_reward += reward
        episode_length += 1
        done = terminated or truncated

        # Store trajectory data
        if save_trajectory:
            # Get current robot state
            ee_pos, ee_orn = env.sim.robot.link_pose()
            base_pos, base_orn = env.sim.robot.link_pose(link_idx=-1)

            trajectory["observations"].append(obs.copy())
            trajectory["actions"].append(action.copy())
            trajectory["rewards"].append(reward)
            trajectory["ee_positions"].append(ee_pos.copy())
            trajectory["ee_orientations"].append(ee_orn.copy())
            trajectory["goal_positions"].append(env.goal_pos.copy())
            trajectory["goal_orientations"].append(env.goal_orn.copy())
            trajectory["base_positions"].append(base_pos.copy())

        # Print progress every 100 steps
        if episode_length % 100 == 0:
            ee_pos, ee_orn = env.sim.robot.link_pose()
            pos_error = np.linalg.norm(ee_pos - env.goal_pos)
            q_dot = np.abs(np.dot(ee_orn, env.goal_orn))
            orn_error = 1.0 - q_dot
            print(
                f"  Step {episode_length}: Reward={reward:.3f}, "
                f"Pos error={pos_error:.3f}, Orn error={orn_error:.3f}"
            )

    elapsed_time = time.time() - start_time

    # Final state
    ee_pos, ee_orn = env.sim.robot.link_pose()
    pos_error = np.linalg.norm(ee_pos - env.goal_pos)
    q_dot = np.abs(np.dot(ee_orn, env.goal_orn))
    orn_error = 1.0 - q_dot
    # Check if goal was actually reached (not just early termination)
    success = (
        terminated
        and pos_error <= env.success_pos_threshold
        and orn_error <= env.success_orn_threshold
    )

    print("  Episode completed:")
    print(f"    Length: {episode_length} steps ({elapsed_time:.2f} seconds)")
    print(f"    Total reward: {episode_reward:.3f}")
    print(f"    Final position error: {pos_error:.3f} m")
    print(f"    Final orientation error: {orn_error:.3f}")
    print(f"    Success: {'Yes' if success else 'No'}")

    results = {
        "episode_num": episode_num,
        "reward": episode_reward,
        "length": episode_length,
        "success": success,
        "pos_error": pos_error,
        "orn_error": orn_error,
        "elapsed_time": elapsed_time,
    }

    if save_trajectory:
        results["trajectory"] = trajectory

    return results


def main():
    """Main evaluation function."""
    args = parse_args()

    # Load configuration
    config = parsing.load_config(args.config)

    # Override GUI setting if specified
    if args.gui:
        config["simulation"]["gui"] = True
        print("GUI visualization enabled")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize environment
    print("Initializing environment...")
    env = SimpleGoalEnv(config)

    # Get dimensions
    obs_shape = env.observation_space.shape
    action_shape = env.action_space.shape
    if obs_shape is None or action_shape is None:
        raise ValueError("Observation or action space shape is None")
    state_dim = obs_shape[0]
    action_dim = action_shape[0]
    action_range = (-1.0, 1.0)

    print(f"State dimension: {state_dim}")
    print(f"Action dimension: {action_dim}")

    # Initialize SAC agent
    rl_config = config.get("rl")
    sac_config = rl_config.get("sac")
    agent = SAC(
        state_dim=state_dim,
        action_dim=action_dim,
        action_range=action_range,
        lr=sac_config.get("lr"),
        gamma=sac_config.get("gamma"),
        tau=sac_config.get("tau"),
        alpha=sac_config.get("alpha"),
        auto_alpha=sac_config.get("auto_alpha"),
        hidden_dim=sac_config.get("hidden_dim"),
        device="cpu",
    )

    # Load checkpoint
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    print(f"\nLoading checkpoint from: {checkpoint_path}")
    agent.load(str(checkpoint_path))
    print("Checkpoint loaded successfully")

    # Set seed if specified
    if args.seed is not None:
        env.reset(seed=args.seed)
        print(f"Using random seed: {args.seed}")

    # Run evaluation episodes
    print(f"\n{'='*80}")
    print(f"Running {args.n_episodes} evaluation episodes")
    print(f"{'='*80}")

    all_results = []
    for episode_num in range(1, args.n_episodes + 1):
        results = evaluate_episode(
            env, agent, episode_num, save_trajectory=args.save_trajectory
        )
        all_results.append(results)

        # Small delay between episodes for visualization
        if args.gui:
            time.sleep(1.0)

    # Print summary
    print(f"\n{'='*80}")
    print("EVALUATION SUMMARY")
    print(f"{'='*80}")

    rewards = [r["reward"] for r in all_results]
    lengths = [r["length"] for r in all_results]
    successes = [r["success"] for r in all_results]
    pos_errors = [r["pos_error"] for r in all_results]
    orn_errors = [r["orn_error"] for r in all_results]

    print(f"\nEpisodes: {args.n_episodes}")
    print(
        f"Success rate: {sum(successes)}/{args.n_episodes} ({100*sum(successes)/args.n_episodes:.1f}%)"
    )
    print("\nRewards:")
    print(f"  Mean: {np.mean(rewards):.3f} ± {np.std(rewards):.3f}")
    print(f"  Min: {np.min(rewards):.3f}, Max: {np.max(rewards):.3f}")
    print("\nEpisode lengths:")
    print(f"  Mean: {np.mean(lengths):.1f} ± {np.std(lengths):.1f}")
    print(f"  Min: {np.min(lengths)}, Max: {np.max(lengths)}")
    print("\nFinal position errors:")
    print(f"  Mean: {np.mean(pos_errors):.3f} ± {np.std(pos_errors):.3f} m")
    print(f"  Min: {np.min(pos_errors):.3f} m, Max: {np.max(pos_errors):.3f} m")
    print("\nFinal orientation errors:")
    print(f"  Mean: {np.mean(orn_errors):.3f} ± {np.std(orn_errors):.3f}")
    print(f"  Min: {np.min(orn_errors):.3f}, Max: {np.max(orn_errors):.3f}")

    # Save results
    if args.save_trajectory:
        import pickle

        results_file = output_dir / "evaluation_results.pkl"
        with open(results_file, "wb") as f:
            pickle.dump(all_results, f)
        print(f"\nSaved trajectory data to: {results_file}")

    # Save summary
    summary_file = output_dir / "evaluation_summary.txt"
    with open(summary_file, "w") as f:
        f.write("EVALUATION SUMMARY\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Checkpoint: {checkpoint_path}\n")
        f.write(f"Config: {args.config}\n")
        f.write(f"Episodes: {args.n_episodes}\n\n")
        f.write(
            f"Success rate: {sum(successes)}/{args.n_episodes} ({100*sum(successes)/args.n_episodes:.1f}%)\n\n"
        )
        f.write(f"Rewards: {np.mean(rewards):.3f} ± {np.std(rewards):.3f}\n")
        f.write(f"Episode lengths: {np.mean(lengths):.1f} ± {np.std(lengths):.1f}\n")
        f.write(
            f"Position errors: {np.mean(pos_errors):.3f} ± {np.std(pos_errors):.3f} m\n"
        )
        f.write(
            f"Orientation errors: {np.mean(orn_errors):.3f} ± {np.std(orn_errors):.3f}\n"
        )

    print(f"\nSaved summary to: {summary_file}")

    env.close()
    print("\nEvaluation complete!")


if __name__ == "__main__":
    main()
