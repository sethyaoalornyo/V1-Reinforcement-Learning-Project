# V2 — Reinforcement Learning Repository
### EECS 590 | DDoS Mitigation as the Foundation Environment

---

## Repository Structure

```
V2/
├── src/
│   ├── mdp/
│   │   └── ddos_mdp.py          ← MDP definition + TabularMDP class
│   ├── classical/
│   │   ├── dp.py                ← Policy Iteration, Value Iteration
│   │   ├── mc.py                ← Monte Carlo methods (4 variants)
│   │   ├── td.py                ← TD(0), n-Step TD, TD(λ) fwd/bwd
│   │   ├── sarsa.py             ← SARSA(0), n-Step, λ fwd/bwd
│   │   └── qlearning.py         ← Q-Learning, Double Q-Learning
│   ├── deep/
│   │   ├── networks.py          ← MLP, DuelingMLP + checkpoint utils
│   │   ├── dqn.py               ← DQN agent (replay + target net)
│   │   └── saliency.py          ← Gradient saliency + Integrated Gradients
│   └── utils/
│       ├── env_wrapper.py       ← Gym-style TabularEnv wrapper
│       ├── replay_buffer.py     ← ReplayBuffer + rotate_replay()
│       └── logger.py            ← TrainingLogger
├── scripts/
│   ├── run_classical.py         ← Train all classical algorithms
│   ├── run_dqn.py               ← Train DQN / Dueling DQN
│   └── manage_replay.py         ← Replay buffer rotation utility
├── checkpoints/
│   └── dqn/ddos/run1/           ← online.pt, target.pt, meta.json
├── replay_buffer/
│   └── ddos/dqn/
│       ├── fresh/buffer.json    ← Current run's experiences
│       └── stale/buffer.json    ← Previous runs (merged by manage_replay)
├── artifacts/                   ← Plots and JSON results per algorithm
├── docs/
│   └── technical-challenges.md  ← Bug log and surprises
├── tests/
│   └── test_v2.py               ← Full test suite
└── requirements.txt
```

---

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run all classical algorithms (saves plots + JSON to artifacts/)
python scripts/run_classical.py --algo all --episodes 3000

# Run a single algorithm
python scripts/run_classical.py --algo sarsa_lambda_bwd --episodes 5000

# Train DQN
python scripts/run_dqn.py --episodes 600

# Train Dueling DQN
python scripts/run_dqn.py --episodes 600 --dueling

# Manage replay buffer (rotate fresh → stale, cap at 50k)
python scripts/manage_replay.py rotate \
    --fresh replay_buffer/ddos/dqn/fresh/buffer.json \
    --stale replay_buffer/ddos/dqn/stale/buffer.json \
    --max_total 50000

# View buffer stats
python scripts/manage_replay.py stats \
    --path replay_buffer/ddos/dqn/fresh/buffer.json

# Run tests
python -m pytest tests/test_v2.py -v
```

---

## Foundation Environment: DDoS Mitigation MDP

The environment models a network controller deciding how to respond to
incoming traffic that may or may not be a DDoS attack.

| Component | Detail |
|---|---|
| **States** | Low (0), Medium (1), High (2), Critical (3), Terminal (4) |
| **Actions** | ALLOW (0), RATE\_LIMIT (1), BLOCK (2) |
| **Discount** | γ = 0.95 |
| **Reward** | Blend of attack-cost and legitimate-traffic-benefit, weighted by p(attack\|state) |
| **Terminal** | State 4 absorbs all transitions (done=True) |

**Why this environment?**
DDoS mitigation is a natural RL problem: the defender cannot observe
the attacker's intent directly (partial observability at the macro
level), must balance false positives (blocking legit users) against
false negatives (allowing attackers through), and acts under stochastic
transition dynamics (network conditions are noisy).  The discrete
action space makes it a natural fit for both tabular methods and DQN.

---

## Classical Algorithms Implemented

### Dynamic Programming (model-based, requires P)
- **Policy Iteration** — evaluate → improve → repeat until stable
- **Value Iteration** — apply Bellman optimality operator until convergence

### Monte Carlo (model-free, episodic)
- **First-Visit MC Prediction** — V(s) using first occurrence per episode
- **Every-Visit MC Prediction** — V(s) using all occurrences
- **On-Policy MC Control** — ε-greedy improvement from first-visit returns
- **Off-Policy MC Control** — Weighted importance sampling

### TD Methods (model-free, bootstrapping)
- **TD(0)** — 1-step bootstrap; simplest TD
- **n-Step TD** — forward view with n-cutoff; n=1 → TD(0), n=∞ → MC
- **TD(λ) Forward** — λ-weighted average of all n-step returns (offline)
- **TD(λ) Backward** — Online eligibility traces (accumulating + replacing)

### SARSA — On-Policy TD Control
- **SARSA(0)** — TD control using the ON-POLICY next action A'
- **n-Step SARSA** — n-cutoff forward view
- **SARSA(λ) Forward** — Offline λ-return version
- **SARSA(λ) Backward** — Online eligibility traces (accumulating + replacing)

### Q-Learning — Off-Policy TD Control
- **Q-Learning** — bootstraps from max Q(s', a') regardless of policy taken
- **Double Q-Learning** — decouples action selection from evaluation to reduce maximisation bias

---

## DRL Algorithm Justification

### Why DQN?

**DQN is the right primary choice for this environment for three reasons:**

**1. Discrete action space.** DQN is specifically designed for environments
with a finite, discrete set of actions — exactly what DDoS mitigation
provides (ALLOW / RATE_LIMIT / BLOCK).  Policy-gradient methods like
PPO and REINFORCE can also handle discrete actions, but DQN's
off-policy nature means it reuses every collected transition (via the
replay buffer), making it dramatically more sample-efficient when
interactions are expensive (e.g., running on real network hardware).

**2. Off-policy learning matches operational requirements.** In a
real deployment, the network operator may need to override the agent's
actions for safety reasons (maintenance windows, manual BLOCK rules,
etc.).  Because DQN separates the behaviour policy (what it does
during training) from the target policy (what it is optimising), this
kind of human-in-the-loop override does not corrupt the learning signal.
On-policy algorithms like PPO and vanilla actor-critic would require
discarding all experience collected under the overridden policy.

**3. Stable training via two key mechanisms.**
- *Experience replay* breaks temporal correlations in consecutive
  transitions, which would otherwise cause catastrophic forgetting or
  oscillation in the neural network.
- *Target network* prevents the bootstrap target from moving at the
  same rate as the prediction, which is the primary cause of
  divergence in naive deep Q-learning.

### Why Dueling DQN as a secondary architecture?

In the DDoS environment, many states have similar value regardless of
which action is taken (e.g., in the "Low" state, ALLOW and RATE_LIMIT
are both fine; the differences are small).  The Dueling architecture
decomposes Q(s, a) = V(s) + A(s, a) − mean_a A(s, a), which lets the
network learn V(s) accurately even from transitions where only one
action was taken.  This stabilises early training and leads to better
V estimates in rarely-visited states.

### Why NOT the other algorithms on this list?

| Algorithm | Reason not primary |
|---|---|
| **REINFORCE** | High-variance gradient estimates; needs many samples to converge. Poor sample efficiency vs DQN. |
| **Vanilla actor-critic** | On-policy; wastes experience from human overrides. Also requires tuning two networks simultaneously. |
| **DDPG** | Designed for CONTINUOUS action spaces. Our actions (ALLOW/RATE\_LIMIT/BLOCK) are discrete — DDPG does not apply without discretisation hacks. |
| **TD3** | Same issue as DDPG — continuous actions only. |
| **PPO** | On-policy; excellent for continuous control and complex environments, but overkill for a 3-action discrete problem and wastes collected data. |
| **TRPO** | Even more complex than PPO (constrained optimisation), on-policy, and the trust region adds hyperparameter burden with no benefit here. |
| **SAC** | Designed for continuous action spaces (entropy-regularised). For discrete actions it requires a modified discrete-SAC variant — not the standard implementation. The entropy bonus is valuable for exploration in large continuous spaces but adds unnecessary complexity here. |

### Summary

DQN is the best-aligned algorithm for this environment because:
- Discrete action space ✓
- Off-policy (handles operator overrides) ✓
- Sample-efficient via replay buffer ✓
- Stable via target network ✓
- Directly interpretable via saliency analysis of the Q-network ✓

The Dueling variant is included as it provides a meaningful improvement
in value estimation at negligible implementation cost.

---

## Checkpoint Organisation

```
checkpoints/
└── dqn/
    └── ddos/
        └── run1/
            ├── online.pt      ← online Q-network weights
            ├── target.pt      ← target Q-network weights (frozen copy)
            └── meta.json      ← {episodes, epsilon, step, architecture}
```

For hyperparameter sweeps, create additional subdirectories:
```
checkpoints/dqn/ddos/run2_lr1e-4/
checkpoints/dqn/ddos/run3_dueling/
```

---

## Replay Buffer Organisation

```
replay_buffer/
└── ddos/
    └── dqn/
        ├── fresh/
        │   └── buffer.json    ← current run (appended by run_dqn.py)
        └── stale/
            └── buffer.json    ← merged history (managed by manage_replay.py)
```

The `manage_replay.py rotate` command merges fresh into stale and
enforces a total-size cap, discarding the oldest experiences when the
buffer exceeds the limit.  This keeps storage manageable and ensures
the agent does not over-train on stale experiences from many runs ago.

---

## Saliency Analysis

Two methods are implemented in `src/deep/saliency.py`:

**Vanilla Gradient Saliency** — `∂Q(s,a)/∂x_i`: fast but can be
noisy because the gradient is evaluated at a single point.

**Integrated Gradients** — integrates the gradient along the path
from an all-zeros baseline to the actual input.  Satisfies the
*completeness* axiom: attributions sum to Q(x) − Q(baseline).
More reliable for understanding which state features the network
truly relies on.

Run `python scripts/run_dqn.py` to generate both heatmaps in
`artifacts/dqn/`.

---

## References

- Sutton & Barto, *Reinforcement Learning: An Introduction*, 2nd ed.
- Mnih et al., "Human-level control through deep reinforcement learning" (DQN, 2015)
- Wang et al., "Dueling Network Architectures for Deep RL" (2016)
- van Hasselt et al., "Deep Reinforcement Learning with Double Q-learning" (2016)
- Sundararajan et al., "Axiomatic Attribution for Deep Networks" (Integrated Gradients, 2017)
