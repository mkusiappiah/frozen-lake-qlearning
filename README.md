# Frozen Lake from First Principles using Q-Learning

This project solves the 8x8 Frozen Lake grid-world with tabular Q-Learning.
The environment, the agent, the training loop, and the evaluation code are
written from scratch in Python. No Gymnasium, Gym, Stable Baselines, RLlib,
or any other reinforcement learning framework is used. The only third party
libraries are NumPy for array math and Matplotlib for plots.

- Name: Michael Kusi-Appiah
- Index Number: 22424580
- Course: DCIT 614 Reinforcement Learning
- Assignment: Programming Assignment 1
- GitHub Repository: https://github.com/mkusiappiah/frozen-lake-qlearning

## 1. Introduction

### What is Reinforcement Learning

Reinforcement Learning (RL) studies how an agent learns to act inside an
environment to maximize a numeric reward. The agent observes a state, takes
an action, receives a reward, and moves to a new state. It repeats this loop
and updates its behavior from experience. RL needs no labeled dataset. The
agent learns from trial, error, and the reward signal alone. The standard
formal model is the Markov Decision Process (MDP), defined by states S,
actions A, a transition function P(s' | s, a), a reward function R, and a
discount factor gamma.

### What is Frozen Lake

Frozen Lake is a grid-world. The agent starts at a Start cell and must reach
a Goal cell while avoiding Holes. Each cell has one type:

- S: Start state
- F: Frozen safe state
- H: Hole, a terminal trap
- G: Goal, the terminal target

The map used here is the 8x8 grid from the assignment:

```
SFFFFFFF
FFFFFFFF
FFFHFFFF
FFFHFFFF
FFFHFFFF
FHHFFFHF
FHFFHFHF
FFFHFFFG
```

The agent must learn a policy that reaches G with the highest probability.

## 2. Environment Design

The environment lives in [environment.py](environment.py) as the class
`FrozenLakeEnv`. It exposes `reset()`, `step(action)`, `render()`,
`get_state()`, and `is_terminal()`.

### State representation

A state is a single integer index. The grid has 8 rows and 8 columns, so
there are 64 states. The index maps to coordinates with:

```
state = row * ncols + col
row, col = divmod(state, ncols)
```

State 0 is the Start at (0, 0). State 63 is the Goal at (7, 7).

### Action representation

Four discrete actions, matching the assignment order:

- 0 = Left
- 1 = Down
- 2 = Right
- 3 = Up

A move that would leave the grid keeps the agent in place. This enforces the
boundary.

### Reward structure

- +1.0 when the agent reaches the Goal
- 0.0 when the agent falls in a Hole
- 0.0 on every other step

This sparse reward matches the classic Frozen Lake definition. The discount
factor gamma propagates the single goal reward back along the path, so states
closer to the goal end up with higher value.

### Stochastic option

The class accepts `is_slippery=True` for Bonus Option A. When slippery, the
intended action runs with probability `intended_prob` and each perpendicular
action runs with probability `(1 - intended_prob) / 2`. Setting
`intended_prob = 1/3` reproduces the classic slippery dynamics where each of
the three outcomes has equal probability.

## 3. Q-Learning Algorithm

Q-Learning is a model-free, off-policy, value-based method. It learns an
action-value function Q(s, a). This value estimates the expected discounted
return from taking action a in state s and acting greedily afterward. The
agent stores Q in a table of shape 64 by 4, one entry per state-action pair.
The table starts at zero. The agent code is in [agent.py](agent.py).

### Update equation

After each transition (s, a, r, s'), the agent applies the tabular
Q-Learning rule:

```
Q(s, a) <- Q(s, a) + alpha * [ r + gamma * max_a' Q(s', a') - Q(s, a) ]
```

- alpha is the learning rate. It sets how much one update shifts the estimate.
- gamma is the discount factor. It sets how much future reward counts now.
- r is the immediate reward.
- max_a' Q(s', a') is the best value available from the next state.
- The bracket term is the temporal difference (TD) error.

When s' is terminal, the future term drops to zero, so the target equals r.
Q-Learning is off-policy because the target uses the max over next actions,
not the action the exploration policy chose.

### Exploration strategy

The agent uses epsilon-greedy action selection.

- With probability epsilon, it picks a random action. This is exploration.
- With probability 1 - epsilon, it picks argmax Q(s, a). This is exploitation.

Epsilon starts high and decays each episode toward a small floor:

```
epsilon <- max(epsilon_min, epsilon * epsilon_decay)
```

Early training explores widely. Late training trusts the learned values. Ties
in the argmax break at random to avoid a fixed bias toward action 0.

## 4. Training Procedure

Training is in [train.py](train.py). Each episode resets the environment,
then loops: select an action, step, apply the Q-update, repeat until the
episode ends or hits the step cap.

### Hyperparameters (deterministic run)

- Episodes: 20000
- Max steps per episode: 200
- Learning rate alpha: 0.1
- Discount factor gamma: 0.99
- Epsilon start: 1.0
- Epsilon min: 0.01
- Epsilon decay: 0.9995 per episode
- Seed: 42

### Recorded statistics

The training loop records, per episode:

- Total episode reward
- Number of steps
- Success flag (1 if the agent reached the Goal)
- Epsilon value
- Running count of successful episodes

It saves the Q-table, a JSON summary, the raw curves, and three plots to the
[results/](results/) folder.

## 5. Results

### Deterministic environment

- Training success rate over 20000 episodes: 89.88 percent
- Greedy evaluation over 100 episodes: 100.00 percent success
- Average reward in evaluation: 1.0000
- Failures: 0 out of 100
- Successful runs: 100 out of 100

The success-rate curve climbs from near zero, passes 90 percent around episode
5000, and settles near 100 percent once epsilon reaches its floor. See
[results/success_rate.png](results/success_rate.png),
[results/reward_curve.png](results/reward_curve.png), and
[results/epsilon_decay.png](results/epsilon_decay.png).

### Learned policy (greedy)

```
 ↓  ↓  ↓  ↓  ↓  ↓  ↓  ↓
 →  →  →  →  →  ↓  ↓  ↓
 ↑  ↑  ↑  H  →  →  →  ↓
 ↑  ↑  ↑  H  ↑  →  →  ↓
 ↑  ↑  ↑  H  →  →  →  ↓
 ↑  H  H  →  ↑  ↑  H  ↓
 ←  H  ←  ←  H  ↑  H  ↓
 ←  ←  ←  H  ←  ←  ←  G
```

From the Start the greedy path is: down once, then right along row 1 to
column 5, down to row 2, right to column 7, then straight down column 7 to the
Goal. This is a 14 step route, which equals the shortest safe path on this
map. The arrows in cells far from this route are not fully tuned. The agent
rarely visits those cells under a near greedy policy, so their values stay
weak. This does not affect performance from the Start, where the policy is
optimal.

### Slippery environment (Bonus Option A)

- Episodes: 50000, epsilon decay 0.9998
- Training success rate: 68.66 percent
- Greedy evaluation over 100 episodes: 84.00 percent success
- Greedy evaluation over 1000 episodes: 88.00 percent success

Perfect success is impossible here. Random slips push the agent off the
intended path, so even an optimal policy fails some episodes. The agent learns
a cautious policy that steers away from holes when slipping is likely. See
[results/slippery_success_rate.png](results/slippery_success_rate.png).

### Exploration comparison (Bonus Option C)

The script [compare_exploration.py](compare_exploration.py) trains two agents
on the deterministic map under the same seed and budget.

- Pure epsilon-greedy (epsilon fixed at 0.1): learns fast but its on-policy
  success rate plateaus near 93 percent, since it keeps taking random actions
  10 percent of the time forever.
- Decaying epsilon-greedy (1.0 to 0.01): starts slower because early epsilon
  is high, then climbs and settles near 100 percent on-policy as exploration
  fades.

Both reach 100 percent in greedy evaluation, since greedy evaluation ignores
epsilon. The plot
[results/exploration_comparison.png](results/exploration_comparison.png) shows
the tradeoff: fixed epsilon gives a quick start but a noisy ceiling, while
decay gives a clean convergence to the optimal route.

### Discussion

- Tabular Q-Learning solves the deterministic 8x8 map to optimality.
- gamma = 0.99 carries the sparse goal reward back across the 14 step path.
- Epsilon decay matters. It balances early exploration with late exploitation.
- Stochastic transitions cap achievable success well below 100 percent and
  need more episodes plus a more cautious policy.

## 6. Execution Instructions

Install dependencies:

```
pip install -r requirements.txt
```

Train the deterministic agent (saves Q-table and plots to results/):

```
python train.py
```

Evaluate the trained agent over 100 episodes:

```
python evaluate.py --episodes 100
```

Render one greedy rollout step by step:

```
python evaluate.py --demo
```

Train and evaluate the slippery version (Bonus A):

```
python train.py --slippery --episodes 50000 --epsilon-decay 0.9998 --prefix slippery_
python evaluate.py --slippery --q-table results/slippery_q_table.npy --episodes 1000
```

Compare exploration strategies (Bonus C):

```
python compare_exploration.py
```

## 7. Repository Structure

```
frozen-lake-qlearning/
├── environment.py          # FrozenLakeEnv from first principles
├── agent.py                # QLearningAgent (Q-table, epsilon-greedy, update)
├── policy.py               # policy extraction and arrow display
├── train.py                # training loop, stats, plots
├── evaluate.py             # greedy evaluation and demo rollout
├── compare_exploration.py  # Bonus C: pure vs decaying epsilon
├── requirements.txt
├── README.md
├── report.pdf
└── results/                # Q-tables, plots, summaries
```

## 8. Bonus Tasks Implemented

- Option A: stochastic slippery transitions in `FrozenLakeEnv`.
- Option B: training plots for reward, success rate, and epsilon.
- Option C: pure versus decaying epsilon-greedy comparison.
