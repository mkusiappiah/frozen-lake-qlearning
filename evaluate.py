"""Evaluation of a trained Frozen Lake Q-Learning agent.

The agent acts greedily with respect to the learned Q-table. The script
reports success rate, average reward, number of failures, and number of
successful runs over a set of episodes.

Usage:
    python evaluate.py
    python evaluate.py --episodes 100 --q-table results/q_table.npy --slippery
"""

import os
import argparse

import numpy as np

from environment import FrozenLakeEnv
from agent import QLearningAgent
from policy import extract_policy, render_policy


RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


def evaluate(agent, env, episodes=100, max_steps=200):
    """Run greedy episodes and return a results dictionary."""
    rewards = []
    successes = 0
    failures = 0

    for _ in range(episodes):
        state = env.reset()
        total_reward = 0.0
        done = False
        steps = 0

        while not done and steps < max_steps:
            action = agent.select_action(state, greedy=True)
            state, reward, done, _ = env.step(action)
            total_reward += reward
            steps += 1

        rewards.append(total_reward)
        if env.cell_type(state) == "G":
            successes += 1
        else:
            failures += 1

    return {
        "episodes": episodes,
        "success_rate_percent": 100.0 * successes / episodes,
        "average_reward": float(np.mean(rewards)),
        "successful_runs": successes,
        "failures": failures,
    }


def demo_episode(agent, env, max_steps=200):
    """Render a single greedy rollout step by step."""
    state = env.reset()
    print("Start:")
    env.render()
    done = False
    steps = 0
    while not done and steps < max_steps:
        action = agent.select_action(state, greedy=True)
        state, reward, done, _ = env.step(action)
        steps += 1
        env.render()
    outcome = "GOAL reached" if env.cell_type(state) == "G" else "fell in a HOLE"
    print(f"Episode ended after {steps} steps: {outcome}.")


def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained Q-Learning agent.")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument(
        "--q-table",
        type=str,
        default=os.path.join(RESULTS_DIR, "q_table.npy"),
        help="path to the saved Q-table",
    )
    parser.add_argument("--slippery", action="store_true")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--demo", action="store_true", help="render one greedy rollout")
    args = parser.parse_args()

    env = FrozenLakeEnv(is_slippery=args.slippery, seed=args.seed)
    agent = QLearningAgent(env.n_states, env.n_actions, seed=args.seed)
    agent.load(args.q_table)

    policy = extract_policy(agent.q_table, env)
    print("Learned policy:")
    print(render_policy(policy, env))
    print()

    results = evaluate(agent, env, episodes=args.episodes, max_steps=args.max_steps)
    print("Evaluation over {} episodes:".format(results["episodes"]))
    print(f"  Success rate    : {results['success_rate_percent']:.2f}%")
    print(f"  Average reward  : {results['average_reward']:.4f}")
    print(f"  Successful runs : {results['successful_runs']}")
    print(f"  Failures        : {results['failures']}")

    if args.demo:
        print("\nDemonstration rollout:")
        demo_episode(agent, env)


if __name__ == "__main__":
    main()
