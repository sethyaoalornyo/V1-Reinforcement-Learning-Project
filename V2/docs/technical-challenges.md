# Technical Challenges & Surprises

Log of bugs encountered, design decisions, and things that were
surprising during the V2 implementation.  Kept as a running list per
the V2 assignment requirements.

---

## Bugs & Issues

### 1. Off-Policy MC — Importance Weights Collapse to Zero
**Problem:** In `mc_control_off_policy_is`, if the greedy action at
any step differs from the behaviour action, the IS weight ρ becomes
zero and the backward sweep terminates immediately.  With a high
exploration rate `ε_b`, most episodes produced zero usable updates
for states beyond the first step.

**Fix:** Lowered the default `epsilon_b` for the behaviour policy to
`0.3` so the greedy action is chosen more often, giving longer usable
tails.  Also verified the weighted IS update rule (using cumulative
weights `C[s][a]`) rather than ordinary IS, which reduces variance.

---

### 2. TD(λ) Forward View — Boundary Term Off-By-One
**Problem:** When computing the λ-return for timestep `t`, the
terminal n-step (where n = T − t) must use the full Monte Carlo return
with NO bootstrapping.  An off-by-one in the loop condition was
accidentally bootstrapping at the final step, producing slightly
inflated V estimates.

**Fix:** Added the explicit check `if n < T - t: (intermediate)` vs
`else: (terminal MC term)` in `td_lambda_forward`.

---

### 3. Eligibility Traces — Forgetting to Reset Per Episode
**Problem:** The eligibility trace vector `e` was initialised once
outside the episode loop, so traces from episode k were bleeding into
episode k+1.  This caused V to diverge for high λ values (≥ 0.9).

**Fix:** Moved `e = [0.0] * env.nS` inside the episode loop.  This
is the standard implementation — traces must be reset at the start of
every episode.

---

### 4. DQN — Target Network Divergence on Small Environments
**Problem:** With `target_update = 5` (very frequent) and a small
replay buffer (capacity < 500), the DQN loss oscillated wildly
because the target network was being updated before the online network
had seen a representative distribution of transitions.

**Fix:** Increased `target_update` to 20 and `buffer_capacity` to
10,000.  Also added gradient clipping (`max_norm=1.0`) in the
Adam optimiser step, which immediately stabilised training.

---

### 5. One-Hot Encoding Gradient Graph — `requires_grad` Placement
**Problem:** In `gradient_saliency`, calling
`state_to_tensor(...).requires_grad_(True)` AFTER creating the tensor
meant the gradient was not tracked through the tensor creation step.
The gradient was `None` after `backward()`.

**Fix:** Call `.requires_grad_(True)` on the tensor BEFORE passing it
through the model, which correctly tells autograd to track gradients
from that point forward.

---

### 6. SARSA(λ) Backward — Inner Loop Complexity
**Surprise:** The naive implementation of the backward eligibility
trace update iterates over ALL (s, a) pairs at every step, making the
inner loop O(nS × nA) per step.  For our 5-state, 3-action environment
this is trivial, but for larger discrete environments (e.g., 1000
states × 10 actions) this would be 10,000 operations per step.

**Alternative for larger environments:** Only update (s, a) pairs
whose trace `e[s][a] > threshold` (sparse trace), which reduces the
inner loop to O(number of recently visited states).

---

### 7. Double Q-Learning — Symmetric Update Required
**Surprise:** In `double_q_learning`, it is important to randomly
choose WHICH table (A or B) to update at each step — not alternating
deterministically (A on even steps, B on odd steps).  Deterministic
alternation caused one table to consistently underestimate because it
was always evaluated by the other table on the "wrong" half of the
trajectory distribution.  Random 50/50 selection equalises this.

---

### 8. `src/mdp_ddos.py` (V1) — Duplicate `if __name__ == "__main__"` Block
**Inherited bug from V1:** `src_mdp_ddos.py` had a nested
`if __name__ == "__main__"` block inside another `if __name__ == ...`
block, which caused an `IndentationError` at import time.  This was
refactored into a clean `main()` function in V2.

---

### 9. Replay Buffer Rotation — Last-Write-Wins Race Condition
**Potential issue:** If two training processes write to the same
`fresh/buffer.json` simultaneously, the second write silently
overwrites the first.  For single-process training (our current setup)
this is fine, but worth noting for future multi-worker setups.

**Mitigation:** Use a lock file or UUID-tagged filenames for each
worker's output, then merge with `manage_replay.py rotate`.

---

## Things That Surprised Me

- **TD(0) vs MC convergence:** TD(0) converges much faster (in terms
  of wall-clock time) than MC because it updates after every step
  rather than waiting for episode completion.  For the DDoS MDP, where
  episodes can last up to 200 steps, MC was noticeably slower.

- **DQN vs tabular Q-learning on a tiny MDP:** On a 5-state
  environment, tabular Q-learning converges to a better policy in 1/10
  the time of DQN.  DQN's real advantage shows up when the state space
  is too large to enumerate — the neural network generalises across
  similar states, which a table cannot do.

- **Dueling DQN advantage:** Even on the tiny MDP, the Dueling
  architecture converges more stably than the plain MLP because it can
  learn the state-value V(s) independently of action advantages.  In
  states where all actions are roughly equal (e.g., "Low" traffic state
  where any action leads to good outcomes), the plain DQN wasted
  capacity trying to distinguish A(s,0), A(s,1), A(s,2) which are all
  near zero.  The Dueling network learns V(s) ≈ high and A ≈ 0 cleanly.

- **Integrated Gradients vs Vanilla Gradients:** For our one-hot
  inputs, vanilla gradient saliency was dominated by whichever input
  dimension happened to have the steepest local gradient — not
  necessarily the one most "responsible" for the output.  Integrated
  gradients gave cleaner attribution that aligned better with human
  intuition (e.g., "High" and "Critical" states correctly attributed
  high importance to those specific state dimensions).
