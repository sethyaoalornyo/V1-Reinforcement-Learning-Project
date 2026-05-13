# Technical Challenges Log

## V1

### Challenge: MDP transition probability normalisation
**Problem:** Early versions of `build_ddos_mdp` had transition probabilities
that did not sum exactly to 1.0 due to floating-point arithmetic. This caused
subtle bugs in policy evaluation where V(s) would not converge to the correct
value.  
**Fix:** Added an assertion in tests that checks `abs(sum(probs) - 1.0) < 1e-9`
for every (s, a) pair. Fixed the transition table manually.

### Challenge: Policy evaluation infinite loop
**Problem:** Policy evaluation with `max_iters=inf` would hang on the terminal
state because V[terminal] was never updated (it was always 0 from `done=True`)
but the convergence check kept running.  
**Fix:** Added `max_iters` cap and confirmed that terminal-state transitions
set `done=True`, which forces `v_next = 0.0` correctly.

---

## V2

### Challenge: DQN divergence with naive Q-target
**Problem:** Without a target network, the Q-network's targets moved every step,
causing oscillation and divergence after ~200 episodes.  
**Fix:** Added a frozen target network updated every `target_update_freq` steps.
This is the standard DQN stabilisation mechanism (Mnih et al., 2015).

### Challenge: One-hot state encoding
**Problem:** The tabular MDP uses integer states (0–4), but PyTorch networks
expect float tensors. Passing raw integers caused silent type errors where the
network received all-zero inputs.  
**Fix:** All state inputs are converted to one-hot vectors of dimension `nS`
before being passed to the network. This is handled consistently in
`run_dqn.py` and the Bayesian tuning objective.

### Challenge: Replay buffer sampling with small buffer
**Problem:** Calling `buffer.sample(64)` when fewer than 64 transitions have
been collected caused `random.choices` to wrap around and sample duplicates,
producing biased gradients early in training.  
**Fix:** Added a guard: `if len(self.buffer) < self.batch_size: return None`.
Learning only begins once the buffer has enough distinct transitions.

---

## V3

### Challenge: Optuna trial pruning
**Problem:** Early Optuna trials with very small `epsilon_decay` values
(< 50) caused the agent to stop exploring immediately, resulting in poor
policies that dominated the surrogate model early in the search.  
**Fix:** Set the lower bound of `epsilon_decay` to 100, ensuring the agent
always has a meaningful exploration phase before exploitation.

### Challenge: Nested `if __name__ == "__main__"` in original mdp.py
**Problem:** The original `src_mdp_ddos.py` had a syntactically invalid
nested `if __name__ == "__main__"` block inside an outer one. Python would
raise a `SyntaxError` on import, breaking all downstream code.  
**Fix:** Removed the nested block entirely. The `__main__` guard is only
needed once per file.

### Challenge: Circular import between dqn.py and tuning module
**Problem:** If `bayesian_tuning.py` imported `DQNAgent` at the top level,
and `dqn.py` imported from `src.tuning`, a circular import would occur at
module load time.  
**Fix:** The import of `DQNAgent` is deferred to inside the `_run_trial`
function in `bayesian_tuning.py`, breaking the cycle.
