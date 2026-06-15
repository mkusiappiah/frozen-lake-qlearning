"""Q-Learning agent implemented from first principles.

The agent stores action values in a table and updates them with the
tabular Q-Learning rule:

    Q(s, a) <- Q(s, a) + alpha * [ r + gamma * max_a' Q(s', a') - Q(s, a) ]

Author: Michael Kusi-Appiah
Course: DCIT 614 - Reinforcement Learning
"""

import random
import numpy as np


class QLearningAgent:
    """Tabular Q-Learning agent with epsilon-greedy exploration.

    Parameters
    ----------
    n_states : int
        Number of discrete states.
    n_actions : int
        Number of discrete actions.
    alpha : float
        Learning rate.
    gamma : float
        Discount factor.
    epsilon : float
        Initial exploration rate.
    epsilon_min : float
        Floor for epsilon.
    epsilon_decay : float
        Multiplicative decay applied once per episode.
    seed : int or None
        Seed for the action selector.
    """

    def __init__(
        self,
        n_states,
        n_actions,
        alpha=0.1,
        gamma=0.99,
        epsilon=1.0,
        epsilon_min=0.01,
        epsilon_decay=0.9995,
        seed=None,
    ):
        self.n_states = n_states
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        self.rng = random.Random(seed)
        # Q-table starts at zero for every state-action pair.
        self.q_table = np.zeros((n_states, n_actions), dtype=np.float64)

    def select_action(self, state, greedy=False):
        """Pick an action with epsilon-greedy exploration.

        Set greedy=True to always take the best known action, which is used
        during evaluation.
        """
        if not greedy and self.rng.random() < self.epsilon:
            return self.rng.randrange(self.n_actions)
        return self._argmax(state)

    def _argmax(self, state):
        """Return the best action for a state, breaking ties at random."""
        row = self.q_table[state]
        best = np.max(row)
        # Random tie break avoids a fixed bias toward action 0.
        candidates = [a for a in range(self.n_actions) if row[a] == best]
        return self.rng.choice(candidates)

    def update(self, state, action, reward, next_state, done):
        """Apply the Q-Learning update for one transition."""
        best_next = 0.0 if done else np.max(self.q_table[next_state])
        td_target = reward + self.gamma * best_next
        td_error = td_target - self.q_table[state, action]
        self.q_table[state, action] += self.alpha * td_error
        return td_error

    def decay_epsilon(self):
        """Reduce epsilon toward its floor. Call once per episode."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    # ----- persistence -----

    def save(self, path):
        """Save the Q-table to a .npy file."""
        np.save(path, self.q_table)

    def load(self, path):
        """Load a Q-table from a .npy file."""
        self.q_table = np.load(path)
