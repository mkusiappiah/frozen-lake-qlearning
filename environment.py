"""Frozen Lake environment implemented from first principles.

No Gymnasium, Gym, Stable Baselines, RLlib, or any RL framework is used.
Only the Python standard library and NumPy.

Author: Michael Kusi-Appiah
Course: DCIT 614 - Reinforcement Learning
"""

import random
import numpy as np


# The 8x8 map defined in the assignment.
DEFAULT_MAP = [
    "SFFFFFFF",
    "FFFFFFFF",
    "FFFHFFFF",
    "FFFHFFFF",
    "FFFHFFFF",
    "FHHFFFHF",
    "FHFFHFHF",
    "FFFHFFFG",
]

# Action constants. Order matches the assignment specification.
LEFT, DOWN, RIGHT, UP = 0, 1, 2, 3
ACTIONS = [LEFT, DOWN, RIGHT, UP]
ACTION_NAMES = {LEFT: "Left", DOWN: "Down", RIGHT: "Right", UP: "Up"}
# Arrow glyphs used for policy display.
ACTION_ARROWS = {LEFT: "←", DOWN: "↓", RIGHT: "→", UP: "↑"}


class FrozenLakeEnv:
    """A grid-world where an agent walks from Start to Goal and avoids Holes.

    State representation uses a single integer index: state = row * ncols + col.
    Helper methods convert between the index and (row, col) coordinates.

    Reward structure (configurable):
        goal_reward when the agent reaches G (default 1.0)
        hole_reward when the agent falls in H (default 0.0)
        step_reward on every non-terminal move (default 0.0)

    Set is_slippery=True for the stochastic version (Bonus Option A). When
    slippery, the intended action runs with probability intended_prob and each
    of the two perpendicular actions runs with probability (1 - intended_prob)/2.
    intended_prob = 1/3 reproduces the classic Gym "slippery" dynamics.
    """

    def __init__(
        self,
        desc=None,
        is_slippery=False,
        intended_prob=1.0 / 3.0,
        goal_reward=1.0,
        hole_reward=0.0,
        step_reward=0.0,
        seed=None,
    ):
        self.desc = list(desc) if desc is not None else list(DEFAULT_MAP)
        self.nrows = len(self.desc)
        self.ncols = len(self.desc[0])
        self.n_states = self.nrows * self.ncols
        self.n_actions = len(ACTIONS)

        self.is_slippery = is_slippery
        self.intended_prob = intended_prob
        self.goal_reward = goal_reward
        self.hole_reward = hole_reward
        self.step_reward = step_reward

        # Random generator for slippery transitions and tie handling.
        self.rng = random.Random(seed)

        # Locate the start cell. Default to (0, 0) if no S is present.
        self.start_state = 0
        for r in range(self.nrows):
            for c in range(self.ncols):
                if self.desc[r][c] == "S":
                    self.start_state = self.to_state(r, c)

        self.state = self.start_state

    # ----- coordinate helpers -----

    def to_state(self, row, col):
        """Convert (row, col) to an integer state index."""
        return row * self.ncols + col

    def to_rc(self, state):
        """Convert an integer state index to (row, col)."""
        return divmod(state, self.ncols)

    def cell_type(self, state):
        """Return the map letter (S, F, H, or G) for a state."""
        r, c = self.to_rc(state)
        return self.desc[r][c]

    # ----- core API -----

    def reset(self):
        """Place the agent at the start state and return that state."""
        self.state = self.start_state
        return self.state

    def get_state(self):
        """Return the current integer state."""
        return self.state

    def is_terminal(self, state=None):
        """Return True if the given state (or current state) is H or G."""
        if state is None:
            state = self.state
        return self.cell_type(state) in ("H", "G")

    def _move(self, state, action):
        """Apply a deterministic action and return the resulting state.

        Movement is clamped at the grid boundary, so an illegal move keeps
        the agent in place.
        """
        row, col = self.to_rc(state)
        if action == LEFT:
            col = max(col - 1, 0)
        elif action == DOWN:
            row = min(row + 1, self.nrows - 1)
        elif action == RIGHT:
            col = min(col + 1, self.ncols - 1)
        elif action == UP:
            row = max(row - 1, 0)
        else:
            raise ValueError(f"Invalid action: {action}")
        return self.to_state(row, col)

    def _sample_action(self, action):
        """Return the action that runs after accounting for slip."""
        if not self.is_slippery:
            return action
        perpendicular = [(action - 1) % 4, (action + 1) % 4]
        side = (1.0 - self.intended_prob) / 2.0
        choices = [action, perpendicular[0], perpendicular[1]]
        weights = [self.intended_prob, side, side]
        return self.rng.choices(choices, weights=weights, k=1)[0]

    def step(self, action):
        """Run one action.

        Returns (next_state, reward, done, info).
        """
        if self.is_terminal(self.state):
            # Episode already ended. Return a no-op transition.
            return self.state, 0.0, True, {"warning": "step after terminal"}

        actual = self._sample_action(action)
        next_state = self._move(self.state, actual)
        self.state = next_state

        kind = self.cell_type(next_state)
        if kind == "G":
            reward, done = self.goal_reward, True
        elif kind == "H":
            reward, done = self.hole_reward, True
        else:
            reward, done = self.step_reward, False

        return next_state, reward, done, {"actual_action": actual}

    # ----- rendering -----

    def render(self, mode="human"):
        """Print the grid with the agent marked by [ ]."""
        ar, ac = self.to_rc(self.state)
        lines = []
        for r in range(self.nrows):
            row_chars = []
            for c in range(self.ncols):
                ch = self.desc[r][c]
                if r == ar and c == ac:
                    row_chars.append(f"[{ch}]")
                else:
                    row_chars.append(f" {ch} ")
            lines.append("".join(row_chars))
        text = "\n".join(lines)
        if mode == "human":
            print(text)
            print()
        return text
