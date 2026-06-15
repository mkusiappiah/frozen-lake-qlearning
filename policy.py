"""Policy extraction and display utilities.

These functions read a trained Q-table and turn it into a greedy policy,
then print that policy on the grid with arrows.

Author: Michael Kusi-Appiah
Course: DCIT 614 - Reinforcement Learning
"""

import numpy as np
from environment import ACTION_ARROWS


def extract_policy(q_table, env):
    """Return the greedy action for every non-terminal state.

    A terminal state (H or G) maps to None because no action is taken there.
    """
    policy = {}
    for state in range(env.n_states):
        if env.is_terminal(state):
            policy[state] = None
        else:
            policy[state] = int(np.argmax(q_table[state]))
    return policy


def render_policy(policy, env):
    """Build a grid string that shows the recommended action per cell.

    Symbols: arrows for moves, H for a hole, G for the goal, S marks start.
    """
    lines = []
    for r in range(env.nrows):
        row_syms = []
        for c in range(env.ncols):
            state = env.to_state(r, c)
            kind = env.cell_type(state)
            if kind == "H":
                sym = "H"
            elif kind == "G":
                sym = "G"
            else:
                action = policy[state]
                sym = ACTION_ARROWS[action] if action is not None else "?"
            row_syms.append(f" {sym} ")
        lines.append("".join(row_syms))
    return "\n".join(lines)


def policy_to_arrows(policy, env):
    """Return a flat list of arrow strings indexed by state (for plots)."""
    out = []
    for state in range(env.n_states):
        kind = env.cell_type(state)
        if kind == "H":
            out.append("H")
        elif kind == "G":
            out.append("G")
        else:
            action = policy[state]
            out.append(ACTION_ARROWS[action] if action is not None else "?")
    return out
