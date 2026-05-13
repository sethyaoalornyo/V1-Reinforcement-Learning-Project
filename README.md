# DDoS Mitigation — Reinforcement Learning Capstone
**EECS 590 | Version 3 (Final)**  
**Author:** [Your Name]

---

## Project Summary

This repository applies reinforcement learning to a DDoS mitigation problem.
A network controller learns to respond to incoming traffic — choosing to ALLOW,
RATE_LIMIT, or BLOCK — based on the observed threat level. The environment is a
5-state, 3-action tabular MDP with stochastic transitions and a reward function
that balances false positives (blocking legitimate users) against false negatives
(allowing attackers through).

**V3 addition:** Bayesian hyperparameter tuning via Optuna, applied to the DQN agent.

---

## Repository Structure

```
V3/
├── src/
│   ├── mdp/
│   │   └── ddos_mdp.py          ← MDP definition (states, actions, transitions)
│   ├── classical/
│   │   └── dp.py                ← Policy Iteration, Value Iteration
│   ├── deep/
│   │   ├── networks.py          ← MLP, DuelingMLP, checkpoint utilities
│   │   └── dqn.py               ← DQN agent (replay buffer + target network)
│   ├── tuning/
│   │   └── bayesian_tuning.py   ← Optuna TPE hyperparameter search [V3]
│   └── utils/
│       ├── env_wrapper.py       ← Gym-style TabularEnv
│       ├── replay_buffer.py     ← ReplayBuffer + rotate_replay
│       └── logger.py            ← TrainingLogger
├── scripts/
│   ├── run_classical.py         ← Train DP algorithms
│   ├── run_dqn.py               ← Train DQN / Dueling DQN
│   └── run_tuning.py            ← Run Bayesian hyperparameter search [V3]
├── checkpoints/
│   └── dqn/ddos/run1/           ← online.pt, target.pt, meta.json
├── artifacts/                   ← Plots and JSON results
├── docs/
│   ├── v3_decisions.md          ← What was implemented and why [V3]
│   ├── technical-challenges.md  ← Bug log and surprises
│   └── citations.md             ← References and AI tool usage
├── tests/
│   └── test_v3.py               ← Full test suite
└── requirements.txt
```

---

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run Policy Iteration (classical baseline)
python scripts/run_classical.py --method policy_iteration

# Train DQN (default hyperparameters)
python scripts/run_dqn.py --episodes 600

# Train Dueling DQN
python scripts/run_dqn.py --episodes 600 --dueling

# [V3] Run Bayesian hyperparameter search (50 trials recommended)
python scripts/run_tuning.py --trials 50 --episodes 300

# [V3] Train DQN with the best tuned hyperparameters
python scripts/run_dqn.py --use-tuned-params

# Run tests
python -m pytest tests/test_v3.py -v
```

---

## Environment

| Component  | Detail                                              |
|------------|-----------------------------------------------------|
| States     | Low(0), Medium(1), High(2), Critical(3), Terminal(4)|
| Actions    | ALLOW(0), RATE_LIMIT(1), BLOCK(2)                   |
| Discount γ | 0.95                                                |
| Reward     | Blend of attack cost and legitimate traffic benefit |
| Terminal   | State 4 absorbs all transitions                     |

---

## V3: Bayesian Hyperparameter Tuning

The key V3 addition is automated hyperparameter optimisation using
[Optuna](https://optuna.org/) with the Tree-structured Parzen Estimator (TPE).

### Why Bayesian vs Grid/Random Search

| Method | Trials needed | Uses prior results? |
|--------|---------------|---------------------|
| Grid   | Exponential   | No                  |
| Random | ~100+         | No                  |
| TPE    | 30–50         | Yes                 |

### Hyperparameters Tuned

- Learning rate (`lr`)
- Network size (`hidden_dim`)
- Replay batch size (`batch_size`)
- Epsilon decay schedule (`epsilon_decay`)
- Target network update frequency (`target_update_freq`)
- Architecture choice (`dueling` or standard MLP)

Results are saved to `artifacts/tuning/best_params.json`.

---

## Algorithms

### Classical (Model-Based)
- **Policy Iteration** — evaluate → greedy improve → repeat
- **Value Iteration** — Bellman optimality operator to convergence

### Deep RL
- **DQN** — experience replay + periodic target network sync
- **Dueling DQN** — Q(s,a) = V(s) + A(s,a) − mean A(s,a)

### V3 Addition
- **Bayesian Hyperparameter Tuning** — Optuna TPE over DQN hyperparameter space

For the full justification of what was and was not implemented, see
[`docs/v3_decisions.md`](docs/v3_decisions.md).

---

## Citations and AI Tool Usage

See [`docs/citations.md`](docs/citations.md) for full references and
a log of AI tool usage during development.

---

## Known Issues

See [`docs/technical-challenges.md`](docs/technical-challenges.md).
