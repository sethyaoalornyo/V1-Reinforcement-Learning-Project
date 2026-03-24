Title: Reinforcement Learning for Real-Time DDoS Attack Detection and Mitigation A Markov Decision Process Formulation  

Seth Yao Alornyo  

 School of Electrical Engineering and Computer Science, University of North Dakota, Grand Forks  
 
Contact: seth.alornyo@ndus.edu  

Course: EECS 590 — Reinforcement Learning  

Semester: Spring 2025  

Current Version: V1 — Dynamic Programming Foundation

Table of Contents  

•	Overview  

•	Motivation  

•	MDP Formulation  

•	Project Versions & Roadmap  

•	Dataset  

•	Results (V1)  

•	Expected Outcomes  

•	Project Timeline  

•	References  


Overview:  
This project develops a Reinforcement Learning (RL) framework for real-time detection and mitigation of Distributed Denial of Service (DDoS) attacks on Software-Defined Networks (SDN). The core idea is to reformulate DDoS detection as a sequential decision-making problem using a Markov Decision Process (MDP), enabling an autonomous agent to learn adaptive, cost-sensitive traffic management strategies without relying on static, manually crafted rules.
The project builds upon a prior supervised learning baseline (XGBoost, 98.3% accuracy on 104,345 SDN flow records) and extends it with a full RL pipeline that considers:
1.Temporal dependencies across traffic flows
2.Asymmetric operational costs (false negatives penalized 2× more than false positives)
3. Intermediate actions (e.g., rate-limiting) beyond binary block/allow

Motivation:
The DDoS Threat Landscape
DDoS attacks remain among the most disruptive threats to networked infrastructure. Cloudflare (2024): ~21.3 million DDoS attempts, a 53% year-over-year increase, including a record 5.6 Tbps attack. Vercara (2024): 16,073% growth in attack volume with 270,405 total detected attacks
These trends expose the inadequacy of static, signature-based defenses and motivate intelligent, adaptive systems.
Limitations of Supervised Learning
The prior XGBoost classifier (98.3% accuracy) has the following gaps in a production setting:
Limitation	Description
No temporal modeling	Each flow classified independently, ignoring sequential relationships
Binary output only	No intermediate actions like rate-limiting
Symmetric cost assumption	False negatives and false positives treated equally
Static policy	Model cannot adapt based on cumulative network-wide feedback
Research Questions
1.	Can a finite MDP accurately model the DDoS detection environment using empirical traffic data?
2.	Do Dynamic Programming solvers (Policy Iteration and Value Iteration) converge to the same optimal policy, validating the MDP formulation?
3.	Can model-free RL agents match or exceed supervised classification accuracy while optimizing asymmetric security costs?
4.	What are the best exploration and approximation strategies for high-dimensional network traffic state spaces?

MDP Formulation:
The DDoS detection task is formalized as a finite MDP M = (S, A, P, R, γ):
State Space (S)
•	8 discriminative flow-level features: byteperflow, pktrate, pairflow, pktperflow, dur, flows, switch, port_no
•	Selected via XGBoost feature importance analysis
•	Continuous features discretized using quantile-based binning (KBinsDiscretizer)
•	Total discrete states: |S| = 57,600
Action Space (A)
Three actions beyond binary classification:
Action	Value	Description
Allow	a = 0	Pass traffic through
Block	a = 1	Drop the flow
Rate-Limit	a = 2	Throttle uncertain traffic
Transition Model (P)
Estimated empirically from sequential structure in the training corpus
Transitions are action-independent (traffic generation does not depend on agent decisions)
Unobserved states receive self-loop transitions: P(s | s) = 1.0
Reward Function (R)
Asymmetric reward prioritizing security with a 2:1 false-negative-to-false-positive penalty ratio:
R(s, a) = P(atk|s) × R_atk(a) + P(ben|s) × R_ben(a)
Discount Factor
γ = 0.95 — effective planning horizon of ~20 time steps, balancing immediate response and long-term stability

Project Versions & Roadmap:
V1 — Dynamic Programming Foundation (Complete)
Implemented:
Fully specified MDP environment (DDoSEnv) with empirically estimated transition and reward matrices
Policy Iteration (PI): Alternates Bellman expectation evaluation and greedy improvement; converges in 2 to 4 iterations
                                    V(s) ← R(s, π(s)) + γ Σ_{s′} P(s′|s, π(s)) V(s′)
                                            π′(s) = argmax_a Q(s, a)
Value Iteration (VI): Applies Bellman optimality backup directly; converges in ~100–200 iterations
Both PI and VI converge to identical optimal policies (confirmed: max|V_PI − V_VI| = 0)
Modular DPAgent with CLI, training pipeline, and YAML-based artifact persistence
All 6-unit tests pass (MDP correctness, DP convergence, PI/VI agreement)

V2: Model-Free RL Extensions (Planned: Weeks 10–12)
Introduces agents that learn without a known transition model, enabling online deployment:
Monte Carlo (MC): Episode-based return estimation without a model
Temporal-Difference (TD): TD(n) and TD(λ) with eligibility traces for bootstrapped updates
On-Policy Control: Sarsa(n) and Sarsa(λ) for Q-value estimation under the current policy
Off-Policy Control: Q-Learning for convergence to optimal policy regardless of behavior policy
Exploration Strategies: ε-greedy, Softmax (Boltzmann), Upper Confidence Bound (UCB)

 V3: Deep RL & Function Approximation (Planned: Weeks 13–14)
Addresses scalability of tabular methods for large/continuous state spaces:
Deep Q-Networks (DQN) for neural function approximation
Potential edge device deployment with real-time traffic injection




Dataset:
SDN-DDoS Dataset
Property	Value
Total records	104,345 flow records
Features	23 features + binary label
Class distribution	 60% normal and 40% attack
Train split	83,476 records (80%, seed=42)
Test split	20,869 records (20%, seed=42)
Split strategy	Stratified

Results (V1):
Test	Status
MDP state/action/transition shape validity	 Pass
Reward matrix correctness	 Pass
Policy Iteration convergence	 Pass
Value Iteration convergence	 Pass
PI/VI policy agreement (max|V_PI − V_VI| = 0)	 Pass
CLI artifact persistence	 Pass

Expected Outcomes
Technical Deliverables
1.Reproducible DDoSEnv MDP environment with empirical transition and reward matrices
2.Validated DP solvers (PI & VI) with policy equivalence
3.Model-free RL agent suite (MC, TD, Sarsa, Q-Learning) with configurable exploration
4.CLI-driven training, evaluation, and artifact pipeline with YAML config support
5.Comprehensive evaluation report across accuracy, F1-score, cumulative reward, and action distribution
Performance Targets
Expected Outcome	Metric / Criterion	Baseline (XGBoost)
Comparable F1-Score	≥ 95%	98.60%
Reward Maximization	Positive cumulative reward	N/A
PI/VI Policy Agreement	max|V_PI − V_VI| = 0	 Achieved (V1)
Model-Free Convergence	TD/MC converges within 1000 episodes	Planned (V2)
Adaptive Cost Weighting	FN penalty > FP (2:1 ratio)	Achieved (V1)

Research Contributions:
A principled MDP formulation for real-time network security, generalizable to other intrusion detection settings
An empirical comparison of planning-based (DP) vs. learning-based (model-free RL) approaches on real SDN traffic
A modular, open-source framework serving as a reproducible foundation for deep RL and multi-agent network security research

Project Timeline:
Phase	Milestone	Deliverable	Target
Phase 1 (V1)	MDP + Dynamic Programming	Policy/Value Iteration, Unit Tests	Complete
Phase 2 (V2)	Model-Free RL	MC, TD, Sarsa, Q-Learning agents	Weeks 10–12
Phase 3 (V3)	Deep RL & Exploration	DQN, ε-greedy, UCB strategies	Weeks 13–14
Phase 4	Evaluation & Report	Final paper, demo, ablation study	Week 15

References:
1.	O. Yoachimik and J. Pacheco, "Record-breaking 5.6 Tbps DDoS attack and global DDoS trends for 2024 Q4," Cloudflare Blog, Jan. 2025.
2.	Vercara (DigiCert), "Annual DDoS report 2024: Trends and insights," 2025.
3.	S. Y. Alornyo, D. Agyapong, J. K. Kibiwott, and E. Kim, "Real-time analysis of DDoS attack detection and mitigation measures using XGBoost machine learning algorithm," Univ. North Dakota, 2025.
4.	R. S. Sutton and A. G. Barto, Reinforcement Learning: An Introduction, 2nd ed. MIT Press, 2018.
5.	P. S. Saini, S. Behal, and S. Bhatia, "Detection of DDoS attacks using machine learning algorithms," in Proc. INDIACom, 2020, pp. 16–21.
6.	N. Moustafa and J. Slay, "A hybrid feature selection for network intrusion detection systems," arXiv:1707.05505, 2017.
