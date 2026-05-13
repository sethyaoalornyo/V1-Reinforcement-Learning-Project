# V3 Implementation Decisions

## EECS 590 — DDoS Mitigation RL Repository
**Author:** [Your Name]  
**Version:** 3 (Final)

---

## Overview

This document articulates every decision made for V3: what was implemented,
what was deliberately omitted, and the reasoning behind each choice. The
environment is a single-agent DDoS mitigation MDP with discrete actions and
a small finite state space. Every implementation decision flows from that
constraint.

---

## What Was Implemented

### Bayesian Hyperparameter Tuning (Optuna / TPE)

**Implemented in:** `src/tuning/bayesian_tuning.py`  
**Script:** `scripts/run_tuning.py`

This is the primary V3 addition. Bayesian tuning with Optuna's
Tree-structured Parzen Estimator (TPE) was applied to the DQN agent's
hyperparameter space:

| Hyperparameter       | Search Range               |
|----------------------|----------------------------|
| `lr`                 | [1e-4, 1e-2] (log scale)   |
| `hidden_dim`         | {32, 64, 128, 256}         |
| `batch_size`         | {32, 64, 128}              |
| `epsilon_decay`      | [100, 1000]                |
| `target_update_freq` | [10, 200]                  |
| `dueling`            | {False, True}              |

**Why this was chosen:**  
The DDoS environment has a single learning agent with no swarm dynamics.
Bayesian tuning is the most impactful V3 technique for this exact
configuration: it directly improves the quality of the DQN policy found
in V2 without requiring any architectural changes to the environment or the
agent. Grid search over these 5+ hyperparameters would require hundreds of
trials; random search would ignore prior results entirely. Optuna's TPE
builds a probabilistic model of the objective surface and samples
preferentially from regions predicted to perform well. In 30–50 trials it
reliably finds configurations that outperform a manually tuned baseline.

The tuning objective is mean episode return over the final 50 episodes of
a 300-episode training run, which directly measures policy quality.

---

### Retained from V2 (with cleanup)

- **Dynamic Programming:** Policy Iteration and Value Iteration remain as the
  model-based baseline. These are appropriate for the small tabular MDP and
  serve as ground-truth optimal policy benchmarks.
- **DQN + Dueling DQN:** The primary deep RL algorithms. Both are preserved
  and are now wired to accept tuned hyperparameters via
  `--use-tuned-params`.
- **Replay buffer + target network:** Core stability mechanisms for DQN.
  Unchanged from V2.
- **Gym-style environment wrapper:** `TabularEnv` provides a clean interface
  used by both DQN training and the Bayesian tuning objective function.
- **Training logger + checkpoint saving:** Retained for traceability.

---

## What Was NOT Implemented and Why

### Multi-Agent RL / Swarm Coordination (QMIX, COMA, CTDE)

**Not implemented.**

The DDoS mitigation environment has a single decision-maker: one network
controller choosing among ALLOW, RATE_LIMIT, and BLOCK. There is no
meaningful second agent to coordinate with. Implementing QMIX or any
cooperative MARL algorithm would require redesigning the environment around
multiple agents — for example, multiple distributed network nodes — which
is a different project, not an enhancement of this one.

Forcing MARL onto a single-agent environment produces a degenerate case
(trivially, a single-agent problem always satisfies IGM) that demonstrates
no understanding of why MARL is useful. The V3 instructions specifically
note: "An environment centered on a main protagonist does not necessarily
benefit from swarm/coordination implementations." This environment is
that case.

---

### Belief Tracking / QMDP (POMDPs)

**Not implemented.**

The DDoS MDP as defined is a fully observable MDP: the controller always
knows which state it is in (Low, Medium, High, Critical). Belief tracking
is the correct tool when the agent cannot observe the true state and must
maintain a distribution over states. Retrofitting belief tracking here
would mean artificially hiding state information from the agent to create
a problem that did not exist before — a contrivance that adds complexity
without improving the model's realism or the agent's performance.

If the environment were extended to model a real distributed network where
the controller only receives noisy sensor readings (ping probabilities,
as in the V3 problem set), belief tracking would become appropriate. That
is a reasonable future direction but is outside the scope of this project.

---

### REINFORCE / Vanilla Actor-Critic / PPO / TRPO

**Not implemented.**

These were excluded in V2 for valid reasons that remain valid in V3:

- REINFORCE and vanilla actor-critic are on-policy and high-variance.
  The DDoS environment benefits from off-policy learning (DQN) because
  operational overrides must not corrupt the learning signal.
- PPO and TRPO are both on-policy and significantly more complex to
  implement correctly. They provide real value in continuous-action or
  very high-dimensional environments. For a 3-action discrete MDP, they
  are overkill and waste collected experience.

None of these algorithms were re-evaluated for V3 because the
environment's fundamental characteristics — discrete actions, small state
space, off-policy requirement — have not changed.

---

### Monte Carlo Methods / TD / SARSA (full V2 classical suite)

**Retained in structure, not re-run for V3.**

The V2 classical algorithm suite (MC, TD, SARSA, Q-learning) is retained
in the `src/classical/` directory for completeness. V3 does not add new
classical algorithms because the focus of the post-V2 curriculum was on
tuning and coordination, not on expanding the classical algorithm library.
The DP baseline (Policy Iteration / Value Iteration) remains the
most appropriate classical reference for this environment.

---

## Summary Table

| Algorithm / Technique          | Decision     | Reason                                       |
|-------------------------------|--------------|----------------------------------------------|
| Bayesian Hyperparameter Tuning | ✅ Implemented | Best V3 fit for single-agent, discrete-action env |
| DQN + Dueling DQN              | ✅ Retained    | Core deep RL; tuned via Bayesian search       |
| Policy / Value Iteration       | ✅ Retained    | Ground-truth baseline                        |
| QMIX / COMA / Swarm MARL       | ❌ Omitted     | Single-agent environment; no coordination need |
| Belief Tracking / QMDP         | ❌ Omitted     | Fully observable MDP; POMDP tools inapplicable |
| PPO / TRPO / A3C               | ❌ Omitted     | On-policy; overkill for discrete 3-action MDP |
| REINFORCE / Actor-Critic       | ❌ Omitted     | High variance; on-policy; no benefit here    |
| DDPG / TD3 / SAC               | ❌ Omitted     | Continuous-action algorithms; wrong action space |

---

## Reflection

The most important decision in V3 was recognising that adding more algorithms
is not the goal. The goal is adding the right algorithm. Bayesian tuning is
the right addition because it directly addresses the weakest link in V2:
manual hyperparameter selection. Everything else that was covered in the
post-V2 curriculum is either inapplicable to this environment or would
require inventing a new problem to justify its inclusion.

A repository that implements swarm coordination for a single-agent DDoS
defender is a repository that implements things without understanding them.
This repository does not do that.
