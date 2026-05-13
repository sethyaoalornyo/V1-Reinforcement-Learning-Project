Topic: DDoS Mitigation — Reinforcement Learning Capstone
EECS 590 Version 3 Final  
Seth Yao Alornyo


Project Overview

This repository applies reinforcement learning to a DDoS mitigation problem.
A network controller learns to respond to incoming traffic —choosing to ALLOW,
RATE_LIMIT, or BLOCK — based on the observed threat level. The environment is a
5-state, 3-action tabular MDP with stochastic transitions and a reward function
that balances false positives (blocking legitimate users) against false negatives
(allowing attackers through).

V3 addition: Bayesian hyperparameter tuning via Optuna, applied to the DQN agent.

Repository Structure

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

 Quick Start

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


The Capstone Project Environment

| Component  | Detail                                              |
|------------|-----------------------------------------------------|
| States     | Low(0), Medium(1), High(2), Critical(3), Terminal(4)|
| Actions    | ALLOW(0), RATE_LIMIT(1), BLOCK(2)                   |
| Discount γ | 0.95                                                |
| Reward     | Blend of attack cost and legitimate traffic benefit |
| Terminal   | State 4 absorbs all transitions                     |


 V3: Bayesian Hyperparameter Tuning

The key V3 addition is automated hyperparameter optimisation using
[Optuna](https://optuna.org/) with the Tree-structured Parzen Estimator (TPE).

Why Bayesian vs Grid/Random Search

| Method | Trials needed | Uses prior results? |
|--------|---------------|---------------------|
| Grid   | Exponential   | No                  |
| Random | ~100+         | No                  |
| TPE    | 30–50         | Yes                 |

Hyperparameters Tuned

- Learning rate (`lr`)
- Network size (`hidden_dim`)
- Replay batch size (`batch_size`)
- Epsilon decay schedule (`epsilon_decay`)
- Target network update frequency (`target_update_freq`)
- Architecture choice (`dueling` or standard MLP)

Results are saved to `artifacts/tuning/best_params.json`.

The Algorithms Used

Classical (Model-Based)
- Policy Iteration — evaluate → greedy improve → repeat
- Value Iteration — Bellman optimality operator to convergence

Deep RL
- DQN — experience replay + periodic target network sync
- Dueling DQN — Q(s,a) = V(s) + A(s,a) − mean A(s,a)

V3 Addition
- Bayesian Hyperparameter Tuning — Optuna TPE over DQN hyperparameter space

For the full justification of what was and was not implemented,Can be found at:
[`docs/v3_decisions.md`](docs/v3_decisions.md).


Citations and AI Tool Usage

Can be found at [`docs/citations.md`](docs/citations.md) for full references and
a log of AI tool usage during development.

Known Issues

Can be found at [`docs/technical-challenges.md`](docs/technical-challenges.md).

The Environment of the Project
Environment based on Distributed Denial of Service Attack Detection and Mitigation 

1. Introduction
This project in reinforcement learning repository describes the decisions made in the implementation of the project. What was decided to be implemented, what was consciously excluded from the implementation and why. The goal is to have transparency and control of the course of the project, not to have full coverage of any possible algorithm.
 The repository implements a wide range of classical and deep reinforcement learning techniques to a DDoS mitigation context, from tabular dynamic programming to Deep Q-Networks with Bayesian hyperparameter optimization. 

2. What has been done 
2.1 Full Classical RL Stack 
The classical algorithm suite has been evaluated on a 40-test suite covering the entire model-free tabular landscape. The following are implemented in the repo: 
Dynamic Programming: Value & Policy Iteration (model-based baseline) 
Monte Carlo Methods: On-Policy and Off-Policy MC Control with Weighted Importance Sampling, First-Visit and Every-Visit Prediction 
TD Methods: TD(0), n-Step TD and TD(λ) in forward (offline) and backward (online eligibility trace) forms 
Variants of SARSA include forward and backward SARSA($\lambda$), n-Step SARSA and SARSA($0$).
 There are two flavors of Q-Learning: Standard Q-Learning and Double Q-Learning (decoupling action selection/evaluation).
 
2.2 Deep RL: DQN and Dueling DQN 
We kept the Deep Q-Network implementation and improved it for the final submission. The MLP and Dueling MLP neural architectures are tested and they are working. Deep RL infrastructure is important and consists of: 
Explore the replay buffer with rotation management (max 50k transitions, fresh → stale). 
Target network updates regularly to maintain bootstrap targets Checkpoints organization with support for hyperparameter sweep subdirectory (online.pt, target.pt, meta.json) Saliency analysis: producing heatmaps in artifacts/dqn/ with vanilla gradient saliency and integrated gradients [2], [5].

2.3 Hyperparameter Tuning by Bayesian Optimization
The most immediately applicable ‘niche’ topic of the second part of the course was added to this project: Bayesian hyperparameter tuning. Random searches lack structure, and grid search would be computationally expensive since the DDoS environment is discrete and small. Bayesian optimization with a Gaussian Process surrogate efficiently explores the hyperparameter space (i.e., epsilon decay schedule, learning rate, target update frequency, replay batch size) with many fewer evaluations [6]. A lightweight wrapper wraps the DQN training loop to log evaluation results according to settings and to use an Expected Improvement acquisition function to select the next candidate. The output is a ranked summary of setups in artifacts/hparam_sweep/. 

2.4 Structural Cleanup and Documentation of the Repository
Technical problems.md: Added debugging notes, e.g., a replay buffer rotation bug found during testing and an edge case for eligibility trace decay. 
README.md: Added references to partners and AI technologies used 
This document, decisions.md, discusses the implementation decisions and the reasoning behind the choice of algorithm. Removed any experimental or auxiliary folders not relevant to the main capstone project from the repository. 
3. Choice of Algorithm and Justification
 The algorithms chosen for this repository were selected based on their suitability for the DDoS Mitigation MDP, which is a single-agent, discrete-action, stochastic environment. We discuss each option, the problem it solves, and why it is the best tool for the job in this case below. 

3.1 Dynamic Programming (Value & Policy Iteration) 
DP techniques provide the baseline to the project based on the model. Exact Bellman operators are tractable because the DDoS MDP transition matrix P is known, and short (5 states, 3 actions). Policy Iteration and Value Iteration are used to find the best policy which is a benchmark for all the model free techniques. For Monte Carlo / TD methods, if you do not have this ground truth, there is no systematic way to check for convergence [1].
 3.2 Monte Carlo Methods (On-policy, Off-policy, First-visit, Every-visit)
In the current approach, Monte Carlo techniques are unbiased value estimators as they learn directly from full episodes without bootstrapping. The DDoS setting satisfies the requirements for MC as it is episodic (Terminal state 4 absorbs all transitions). We use First-Visit and Every-Visit MC to show the tradeoff in estimate variance. Off-Policy MC with weighted importance sampling is included because the DDoS use case encourages a behavior/target policy split, where a safe exploratory policy gathers data while the target policy is aggressively optimized, like real network operations where the production policy cannot be unduly exploratory [1].
 3.3 Forward and Backward TD(0), n-Step TD and TD(λ)
TD methods bridge the gap between MC (no bootstrapping) and DP (full bootstrapping). All subsequent TD-based control algorithms are based on the simplest online update, TD(0). The bias-variance tradeoff can be explicitly controlled using n-Step TD: n=1 retrieves TD(0) and n=∞ retrieves MC. The n-step returns are combined in a geometric way in order to cover the whole spectrum. The forward (offline $\lambda$-return) and backward (online eligibility trace) views are constructed to demonstrate their mathematical equivalency, and the backward view is provided as a practical online method [1]. 
3.4 SARSA Variants (n-Step, λ Forward and Backward, SARSA(0)) 
SARSA is the on-policy version of TD control. When we update Q(s, a), we use the action actually taken according to the current policy, not the greedy action. In the DDoS scenario, on-policy control is relevant when the agent strictly follows its ε-greedy decisions during the exploratory training phase. A DDoS scenario, where the impact of a RATE_LIMIT action might take many transitions to manifest, as attack traffic builds up. SARSA(λ) with eligibility traces propagates credit assignment backwards through recent state-action pairs [1]. 
3.5 Q-Learning and Double Q-Learning 
The baseline for off-policy TD control is Q-Learning. Q-Learning is more aggressive in value estimation than SARSA because it bootstraps from max Q(s', a') regardless of the policy used. This is the natural tabular precursor to DQN. In the DDoS case, the overestimation of the value of BLOCK (a large instant reward for preventing attacks) can lead the agent to block legitimate traffic excessively, hence Double Q-Learning is used especially to counteract the maximizing bias. Double Q-Learning decouples the action evaluation and action selection using two value tables, which reduces this bias and produces better calibrated Q-values [1], [4]. 
3.6 MLP Architecture for Deep Q-Networks (DQN)
 There are three reasons for DQN being the main deep reinforcement learning algorithm in this setting. First, DQN is designed specifically for discrete action space (ALLOW, RATE_LIMIT, BLOCK). Second, DQN is off-policy, meaning that the transitions are stored in a replay buffer and reused multiple times, which is crucial when the real network overhead is modeled by environment interactions. Third, two stabilizing mechanisms make DQN tractable where naïve deep Q-learning diverges: the target network prevents the bootstrap target from moving at the same rate as the online network and experience replay breaks the temporal connection between successive transitions [2]. 

3.7 Dueling DQN Architecture 
The Dueling architecture separates action advantage from state value by decomposing $Q(s, a) = V(s) + A(s, a) - \text{mean} A(s, a)$. This is driven directly by the nature of the DDoS environment: in the Low-traffic condition, Q-value variations are minimal, and ALLOW and RATE_LIMIT are almost equal. When only one action is captured each transition, typical MLP cannot learn V(s) well. The Dueling MLP learns V ( s ) from each transition whatever action is chosen . This helps to stabilize early training and improve value estimates in rarely seen states , e.g. Critical ( state 3 ) [3]. 
3.8 Bayesian hyperparameters tuning
Bayesian optimization is a natural choice over grid search and random search for the DDoS training problem, given the expensive evaluation of the objective function (final evaluation return) for each configuration and the continuous hyperparameter space (epsilon-decay rate, learning rate, target update frequency, replay batch size). Once a Gaussian Process surrogate has been used to model the objective surface from observed evaluations, the next candidate is selected using an acquisition function called Expected Improvement, to reduce the number of full training runs needed. In a single protagonist setting like this, where sample efficiency is key, Bayesian tweaking is immediately beneficial [6]. 
4. Repository structural preparedness 
The repository is structured for readability and reuse. Anyone reproducing it should be able to run the code and understand the design choices without any prior context: 
4.1 Structure of Directory 
project/src/ mdp/ddos_mdp.py classical/ (dp, mc, td, sarsa, qlearning) deep/ (networks, dqn, saliency) utils/ (env_wrapper, replay_buffer, logger) 
Scripts/checkpoints/dqn/ddos/run1/replay_buffer/ddos/dqn/{fresh,stale}/artifacts/ (plots, JSON, hparam_sweep/)
 Docs/technical-challenges.md Decisions.md (this document) tests/test_v3.py README.md 
4.2 Datos
 The README.md file includes quick-start instructions, a description of the environment, a synopsis of the algorithm reasoning, and citations for all collaborators and AI tools used in development. Technical problems.md: Complete bug log with root-cause analysis for issues identified, maintained through final delivery choices.md: This document describes implementation choices per the V3 standard. Requirements.txt: pinned dependencies in reproducible copies 

Reward

The project consider a reinforcement learning (RL) policy found by policy iteration for a risk-sensitive control setting. The system is modeled as a finite Markov Decision Process (MDP) with discrete states representing increasing levels of system load or threat severity. The learned policy assigns each state an optimal action: ALLOW, RATE_LIMIT or BLOCK, based on long-term expected return. The resulting state-value function provides quantitative insight into the desirability of each system state under the optimal policy. We model the system as a finite Markov Decision Process (MDP) with discrete states representing increasing levels of system load or threat severity . The learned policy is a mapping from each state to an optimal action (ALLOW, RATE_LIMIT or BLOCK) that maximizes the long-term expected return. The corresponding state-value function quantitatively offers insight into the desirability of each system state under the optimal policy.

5. Prepared for use
 5.1 Testing Set 
The repo also contains a 40-test suite (test_v3.py) to cover all classical algorithms, DQN training loop, replay buffer rotation, checkpoint save/load, and saliency output shapes. No test run suffers from any interpreter or compiler bugs. 
5.2 Limitations Identified
The following restrictions are noted, rather than being quietly ignored: 
The lightweight GP implementation used for Bayesian hyperparameter tuning works, but it is not suitable for wide sweep spaces. For production it would be good to integrate with Ray Tune or Optuna. 
The DDoS MDP is intentionally minimalist (5 states, 3 actions) to create a clean educational environment. This is also reflected in the trained rules that converge quickly but do not generalize to broader network topologies.
Saliency heatmaps are made after training, they are not used to direct training. They are informative.6 A future extension could use attribution-guided exploration. 

6. References and Acknowledgements

The following tools, partners and resources were used to create this repository: 
6. 1 References  
[1] R. S. Sutton and A. G. Barto, Reinforcement Learning: An Introduction, 2nd ed. Cambridge, MA, USA: MIT Press, 2018.
[2] V. Mnih et al., “Human-level control through deep reinforcement learning,” Nature, vol. 518, no. 7540, pp. 529–533, Feb. 2015, doi: 10.1038/nature14236.’
[3] Z. Wang, T. Schaul, M. Hessel, H. van Hasselt, M. Lanctot, and N. de Freitas, “Dueling network architectures for deep reinforcement learning,” in Proc. 33rd Int. Conf. Machine Learning (ICML), New York, NY, USA, Jun. 2016, pp. 1995–2003.
[4] H. van Hasselt, A. Guez, and D. Silver, “Deep reinforcement learning with double Q-learning,” in Proc. 30th AAAI Conf. Artificial Intelligence (AAAI), Feb. 2016, pp. 2094–2100, doi: 10.1609/aaai.v30i1.10295.
[5] M. Sundararajan, A. Taly, and Q. Yan, “Axiomatic attribution for deep networks,” in Proc. 34th Int. Conf. Machine Learning (ICML), Sydney, NSW, Australia, Aug. 2017, pp. 3319–3328.
[6] J. Snoek, H. Larochelle, and R. P. Adams, “Practical Bayesian optimization of machine learning algorithms,” in Advances in Neural Information Processing Systems (NeurIPS), vol. 25, Lake Tahoe, NV, USA, Dec. 2012, pp. 2951–2959.
[7] A. Kazin, “Ddos sdn dataset,” Kaggle dataset, 2022, accessed: 2025-09-02. [Online]. Available: https://www.kaggle.com/datasets/ aikenkazin/ddos-sdn-dataset
6.2 AI Resources Used 
This repository was built with the help of an AI chat bot. The following illustrates how these techniques were employed to promote transparency: 
Claude (Anthropic): Used for exploring edge cases in eligibility trace updates, drafting docstrings, and debugging issues with Python implementation. The author wrote and understood every algorithm logic. 
Checked the Double Q-Learning maximization bias argument and clarified the Integrated Gradients completeness axiom by consulting an AI system built by a team of inventors at Amazon. 
The RL algorithms were primarily implemented without the help of any AI tools.


