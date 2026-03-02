"""Analyze training metrics from training_metrics.npz file.

Command:

```bash
cd /home/federico/catkin_ws/src/mobile_manipulation
python3 mm_rl/analyze_training.py --metrics logs/training_metrics.npz --output logs
```
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_metrics(metrics_path):
    """Load metrics from npz file."""
    data = np.load(metrics_path)
    metrics = {key: np.array(data[key]) for key in data.keys()}
    return metrics


def print_summary(metrics):
    """Print summary statistics of metrics."""
    print("\n" + "=" * 80)
    print("TRAINING METRICS SUMMARY")
    print("=" * 80)

    if "step" in metrics:
        steps = metrics["step"]
        print(f"\nTotal training updates: {len(steps)}")
        print(f"Step range: {steps.min():.0f} - {steps.max():.0f}")

    print("\nAvailable metrics:")
    for key in sorted(metrics.keys()):
        values = metrics[key]
        if len(values) > 0:
            print(f"\n  {key}:")
            print(f"    Shape: {values.shape}")
            print(f"    Min: {values.min():.6f}")
            print(f"    Max: {values.max():.6f}")
            print(f"    Mean: {values.mean():.6f}")
            print(f"    Std: {values.std():.6f}")
            if len(values) > 1:
                # Show trend (last 10% vs first 10%)
                n = len(values)
                first_10 = values[: n // 10].mean()
                last_10 = values[-n // 10 :].mean()
                change = last_10 - first_10
                change_pct = (change / abs(first_10) * 100) if first_10 != 0 else 0
                print(
                    f"    Trend: {change:+.6f} ({change_pct:+.2f}%) [first 10%: {first_10:.6f}, last 10%: {last_10:.6f}]"
                )


def plot_metrics(metrics, output_dir=None):
    """Create plots for all metrics."""
    if "step" not in metrics:
        print("Warning: 'step' not found in metrics, cannot create time-series plots")
        return

    steps = metrics["step"]

    # Determine number of subplots needed
    metric_keys = [k for k in metrics.keys() if k != "step"]
    n_metrics = len(metric_keys)

    if n_metrics == 0:
        print("No metrics to plot")
        return

    # Create figure with subplots
    n_cols = 2
    n_rows = (n_metrics + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
    if n_metrics == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    for idx, key in enumerate(metric_keys):
        ax = axes[idx]
        values = metrics[key]

        # Plot raw values
        ax.plot(steps, values, alpha=0.6, linewidth=0.5, label=key)

        # Add moving average
        if len(values) > 100:
            window = min(100, len(values) // 10)
            moving_avg = np.convolve(values, np.ones(window) / window, mode="valid")
            moving_steps = steps[window - 1 :]
            ax.plot(moving_steps, moving_avg, linewidth=2, label=f"{key} (MA)")

        ax.set_xlabel("Training Step")
        ax.set_ylabel(key)
        ax.set_title(f"{key} over Training")
        ax.legend()
        ax.grid(True, alpha=0.3)

    # Hide unused subplots
    for idx in range(n_metrics, len(axes)):
        axes[idx].axis("off")

    plt.tight_layout()

    if output_dir:
        output_path = Path(output_dir) / "training_metrics_plots.png"
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"\nSaved plots to: {output_path}")
    else:
        plt.show()


def analyze_training_progress(metrics):
    """Analyze training progress and provide insights."""
    print("\n" + "=" * 80)
    print("TRAINING PROGRESS ANALYSIS")
    print("=" * 80)

    if "step" not in metrics:
        print("Cannot analyze progress without 'step' information")
        return

    steps = metrics["step"]
    n = len(steps)

    # Analyze each metric
    for key in sorted(metrics.keys()):
        if key == "step":
            continue

        values = metrics[key]
        if len(values) < 2:
            continue

        print(f"\n{key.upper()}:")

        # Early vs late training
        early_idx = n // 4
        late_idx = -n // 4

        early_mean = values[:early_idx].mean()
        late_mean = values[late_idx:].mean()

        print(f"  Early training (first 25%): {early_mean:.6f}")
        print(f"  Late training (last 25%):   {late_mean:.6f}")
        print(f"  Change: {late_mean - early_mean:+.6f}")

        # Check for convergence
        if len(values) > 100:
            last_100 = values[-100:]
            std_last_100 = last_100.std()
            mean_last_100 = last_100.mean()
            cv = (
                std_last_100 / abs(mean_last_100)
                if mean_last_100 != 0
                else float("inf")
            )

            if cv < 0.1:
                print(f"  Status: Appears converged (CV={cv:.4f} < 0.1)")
            elif cv < 0.3:
                print(f"  Status: Stabilizing (CV={cv:.4f})")
            else:
                print(f"  Status: Still varying (CV={cv:.4f})")

        # Check for improvement
        if key in ["q_value"]:
            if late_mean > early_mean:
                print("  Trend: ✓ Improving (increasing)")
            else:
                print("  Trend: ✗ Declining")
        elif key in ["critic_loss", "actor_loss"]:
            if late_mean < early_mean:
                print("  Trend: ✓ Improving (decreasing)")
            else:
                print("  Trend: ✗ Not improving")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Analyze training metrics from training_metrics.npz"
    )
    parser.add_argument(
        "--metrics",
        type=str,
        default="logs/training_metrics.npz",
        help="Path to training_metrics.npz file",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Directory to save plots (if not specified, plots are displayed)",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip generating plots",
    )

    args = parser.parse_args()

    # Load metrics
    metrics_path = Path(args.metrics)
    if not metrics_path.exists():
        print(f"Error: Metrics file not found: {metrics_path}")
        print("Please provide the correct path to training_metrics.npz")
        sys.exit(1)

    print(f"Loading metrics from: {metrics_path}")
    metrics = load_metrics(metrics_path)

    # Print summary
    print_summary(metrics)

    # Analyze progress
    analyze_training_progress(metrics)

    # Create plots
    if not args.no_plots:
        print("\n" + "=" * 80)
        print("GENERATING PLOTS")
        print("=" * 80)
        plot_metrics(metrics, output_dir=args.output)

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
