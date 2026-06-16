"""Bonus Option C: compare pure epsilon-greedy with decaying epsilon-greedy.

Pure epsilon-greedy holds epsilon fixed for the whole run. Decaying
epsilon-greedy starts high and shrinks toward a small floor. The script
trains one agent of each type under the same seed and budget, then plots
their success-rate curves and reports final greedy evaluation results.

Usage:
    python compare_exploration.py
"""

import os

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from train import train, moving_average, RESULTS_DIR
from evaluate import evaluate
from environment import FrozenLakeEnv


def run_comparison(episodes=20000, seed=42, fixed_epsilon=0.1):
    """Train both strategies and return their stats and eval results."""
    # Pure epsilon-greedy: fixed epsilon, no decay (decay factor 1.0).
    agent_fixed, env, stats_fixed = train(
        episodes=episodes,
        epsilon=fixed_epsilon,
        epsilon_min=fixed_epsilon,
        epsilon_decay=1.0,
        is_slippery=False,
        seed=seed,
        verbose=False,
    )

    # Decaying epsilon-greedy: start at 1.0, decay toward 0.01.
    agent_decay, env2, stats_decay = train(
        episodes=episodes,
        epsilon=1.0,
        epsilon_min=0.01,
        epsilon_decay=0.9995,
        is_slippery=False,
        seed=seed,
        verbose=False,
    )

    eval_env = FrozenLakeEnv(is_slippery=False, seed=999)
    res_fixed = evaluate(agent_fixed, eval_env, episodes=100)
    res_decay = evaluate(agent_decay, eval_env, episodes=100)

    return stats_fixed, stats_decay, res_fixed, res_decay


def plot_comparison(stats_fixed, stats_decay, window=100):
    """Overlay the two success-rate curves and save the figure."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    ma_fixed = moving_average(stats_fixed["episode_success"], window) * 100.0
    ma_decay = moving_average(stats_decay["episode_success"], window) * 100.0

    plt.figure(figsize=(8, 4))
    plt.plot(range(len(ma_fixed)), ma_fixed, label="Pure epsilon-greedy (eps=0.1)")
    plt.plot(range(len(ma_decay)), ma_decay, label="Decaying epsilon-greedy")
    plt.xlabel("Episode")
    plt.ylabel(f"Success rate % (window={window})")
    plt.title("Pure vs decaying epsilon-greedy")
    plt.ylim(-5, 105)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "exploration_comparison.png"), dpi=120)
    plt.close()


def main():
    stats_fixed, stats_decay, res_fixed, res_decay = run_comparison()
    plot_comparison(stats_fixed, stats_decay)

    print("Greedy evaluation over 100 episodes after training:\n")
    print("Pure epsilon-greedy (epsilon fixed at 0.1):")
    print(f"  success rate {res_fixed['success_rate_percent']:.2f}% | "
          f"avg reward {res_fixed['average_reward']:.4f}")
    print("Decaying epsilon-greedy (1.0 -> 0.01):")
    print(f"  success rate {res_decay['success_rate_percent']:.2f}% | "
          f"avg reward {res_decay['average_reward']:.4f}")
    print(f"\nComparison plot saved to {RESULTS_DIR}/exploration_comparison.png")


if __name__ == "__main__":
    main()
