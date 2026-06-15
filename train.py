"""Training loop for the Frozen Lake Q-Learning agent.

Run this file to train an agent, save the Q-table, record statistics,
print the learned policy, and write performance plots (Bonus Option B).

Usage:
    python train.py
    python train.py --episodes 20000 --alpha 0.1 --gamma 0.99 --slippery

Author: Michael Kusi-Appiah
Course: DCIT 614 - Reinforcement Learning
"""

import os
import json
import argparse

import numpy as np
import matplotlib

matplotlib.use("Agg")  # write figures to file without a display
import matplotlib.pyplot as plt

from environment import FrozenLakeEnv
from agent import QLearningAgent
from policy import extract_policy, render_policy


RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


def moving_average(values, window):
    """Return the simple moving average of a list with the given window."""
    if len(values) < window:
        window = max(1, len(values))
    arr = np.asarray(values, dtype=np.float64)
    kernel = np.ones(window) / window
    return np.convolve(arr, kernel, mode="valid")


def train(
    episodes=20000,
    max_steps=200,
    alpha=0.1,
    gamma=0.99,
    epsilon=1.0,
    epsilon_min=0.01,
    epsilon_decay=0.9995,
    is_slippery=False,
    seed=42,
    verbose=True,
):
    """Train one agent and return (agent, env, stats)."""
    env = FrozenLakeEnv(is_slippery=is_slippery, seed=seed)
    agent = QLearningAgent(
        n_states=env.n_states,
        n_actions=env.n_actions,
        alpha=alpha,
        gamma=gamma,
        epsilon=epsilon,
        epsilon_min=epsilon_min,
        epsilon_decay=epsilon_decay,
        seed=seed,
    )

    episode_rewards = []
    episode_steps = []
    episode_success = []   # 1 if the agent reached the goal, else 0
    epsilon_history = []
    successful_episodes = 0

    for ep in range(episodes):
        state = env.reset()
        total_reward = 0.0
        steps = 0
        done = False

        while not done and steps < max_steps:
            action = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)
            agent.update(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            steps += 1

        reached_goal = env.cell_type(state) == "G"
        if reached_goal:
            successful_episodes += 1

        episode_rewards.append(total_reward)
        episode_steps.append(steps)
        episode_success.append(1 if reached_goal else 0)
        epsilon_history.append(agent.epsilon)

        agent.decay_epsilon()

        if verbose and (ep + 1) % 2000 == 0:
            recent = np.mean(episode_success[-2000:]) * 100.0
            print(
                f"Episode {ep + 1:>6} | "
                f"epsilon {agent.epsilon:6.3f} | "
                f"success last 2000 {recent:6.2f}% | "
                f"total successes {successful_episodes}"
            )

    stats = {
        "episode_rewards": episode_rewards,
        "episode_steps": episode_steps,
        "episode_success": episode_success,
        "epsilon_history": epsilon_history,
        "successful_episodes": successful_episodes,
        "episodes": episodes,
        "hyperparameters": {
            "alpha": alpha,
            "gamma": gamma,
            "epsilon_start": epsilon,
            "epsilon_min": epsilon_min,
            "epsilon_decay": epsilon_decay,
            "max_steps": max_steps,
            "is_slippery": is_slippery,
            "seed": seed,
        },
    }
    return agent, env, stats


def save_plots(stats, prefix=""):
    """Write reward, success-rate, and epsilon plots to the results folder."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    window = 100

    rewards = stats["episode_rewards"]
    success = stats["episode_success"]
    eps = stats["epsilon_history"]

    # 1. Moving average of episode reward.
    plt.figure(figsize=(8, 4))
    ma = moving_average(rewards, window)
    plt.plot(range(len(ma)), ma)
    plt.xlabel("Episode")
    plt.ylabel(f"Reward (moving average, window={window})")
    plt.title("Episode reward during training")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, f"{prefix}reward_curve.png"), dpi=120)
    plt.close()

    # 2. Moving average of success rate.
    plt.figure(figsize=(8, 4))
    ma_s = moving_average(success, window) * 100.0
    plt.plot(range(len(ma_s)), ma_s, color="green")
    plt.xlabel("Episode")
    plt.ylabel(f"Success rate % (window={window})")
    plt.title("Success rate during training")
    plt.ylim(-5, 105)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, f"{prefix}success_rate.png"), dpi=120)
    plt.close()

    # 3. Epsilon decay.
    plt.figure(figsize=(8, 4))
    plt.plot(range(len(eps)), eps, color="orange")
    plt.xlabel("Episode")
    plt.ylabel("Epsilon")
    plt.title("Exploration rate (epsilon) over time")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, f"{prefix}epsilon_decay.png"), dpi=120)
    plt.close()


def save_artifacts(agent, stats, prefix=""):
    """Save the Q-table and a JSON summary of the run."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    agent.save(os.path.join(RESULTS_DIR, f"{prefix}q_table.npy"))

    # Store a compact JSON summary plus the raw curves for later inspection.
    summary = {
        "hyperparameters": stats["hyperparameters"],
        "episodes": stats["episodes"],
        "successful_episodes": stats["successful_episodes"],
        "train_success_rate_percent": 100.0
        * stats["successful_episodes"]
        / stats["episodes"],
    }
    with open(os.path.join(RESULTS_DIR, f"{prefix}summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    np.savez_compressed(
        os.path.join(RESULTS_DIR, f"{prefix}training_curves.npz"),
        episode_rewards=np.asarray(stats["episode_rewards"]),
        episode_steps=np.asarray(stats["episode_steps"]),
        episode_success=np.asarray(stats["episode_success"]),
        epsilon_history=np.asarray(stats["epsilon_history"]),
    )


def main():
    parser = argparse.ArgumentParser(description="Train a Frozen Lake Q-Learning agent.")
    parser.add_argument("--episodes", type=int, default=20000)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--epsilon", type=float, default=1.0)
    parser.add_argument("--epsilon-min", type=float, default=0.01)
    parser.add_argument("--epsilon-decay", type=float, default=0.9995)
    parser.add_argument("--slippery", action="store_true", help="stochastic transitions")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--prefix", type=str, default="", help="filename prefix for outputs")
    args = parser.parse_args()

    agent, env, stats = train(
        episodes=args.episodes,
        max_steps=args.max_steps,
        alpha=args.alpha,
        gamma=args.gamma,
        epsilon=args.epsilon,
        epsilon_min=args.epsilon_min,
        epsilon_decay=args.epsilon_decay,
        is_slippery=args.slippery,
        seed=args.seed,
    )

    save_artifacts(agent, stats, prefix=args.prefix)
    save_plots(stats, prefix=args.prefix)

    policy = extract_policy(agent.q_table, env)
    print("\nLearned policy:")
    print(render_policy(policy, env))

    train_sr = 100.0 * stats["successful_episodes"] / stats["episodes"]
    print(f"\nTraining success rate: {train_sr:.2f}%")
    print(f"Q-table and plots saved in: {RESULTS_DIR}")


if __name__ == "__main__":
    main()
