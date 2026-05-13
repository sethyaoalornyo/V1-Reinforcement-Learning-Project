from __future__ import annotations
import random
from typing import Optional

import torch
import torch.nn as nn
import torch.optim as optim

from src.deep.networks import MLP, DuelingMLP
from src.utils.replay_buffer import ReplayBuffer


class DQNAgent:
    """
    Deep Q-Network agent with:
      - Experience replay
      - Periodically updated target network
      - Epsilon-greedy exploration with linear decay
    """

    def __init__(
        self,
        nS: int,
        nA: int,
        hidden_dim: int        = 128,
        lr: float              = 1e-3,
        gamma: float           = 0.95,
        epsilon_start: float   = 1.0,
        epsilon_end: float     = 0.05,
        epsilon_decay: int     = 500,
        batch_size: int        = 64,
        target_update_freq: int = 50,
        buffer_capacity: int   = 50_000,
        dueling: bool          = False,
        seed: Optional[int]    = None,
    ) -> None:
        if seed is not None:
            torch.manual_seed(seed)
            random.seed(seed)

        self.nA              = nA
        self.gamma           = gamma
        self.epsilon         = epsilon_start
        self.epsilon_end     = epsilon_end
        self.epsilon_decay   = epsilon_decay
        self.batch_size      = batch_size
        self.target_update_freq = target_update_freq
        self._step           = 0

        arch = DuelingMLP if dueling else MLP
        self.online = arch(nS, hidden_dim, nA)
        self.target = arch(nS, hidden_dim, nA)
        self.target.load_state_dict(self.online.state_dict())
        self.target.eval()

        self.optimizer = optim.Adam(self.online.parameters(), lr=lr)
        self.loss_fn   = nn.SmoothL1Loss()
        self.buffer    = ReplayBuffer(capacity=buffer_capacity, seed=seed)

    # ── action selection ──────────────────────────────────────────────────────

    def select_action(self, state: int) -> int:
        self.epsilon = max(
            self.epsilon_end,
            self.epsilon - (1.0 - self.epsilon_end) / self.epsilon_decay,
        )
        if random.random() < self.epsilon:
            return random.randint(0, self.nA - 1)
        s_tensor = torch.zeros(self._nS_hint if hasattr(self, "_nS_hint") else 5)
        s_tensor[state] = 1.0
        with torch.no_grad():
            return int(self.online(s_tensor.unsqueeze(0)).argmax(dim=1).item())

    def select_action_onehot(self, state_vec: torch.Tensor) -> int:
        if random.random() < self.epsilon:
            return random.randint(0, self.nA - 1)
        with torch.no_grad():
            return int(self.online(state_vec.unsqueeze(0)).argmax(dim=1).item())

    # ── learning step ─────────────────────────────────────────────────────────

    def push(self, s: int, a: int, r: float, s2: int, done: bool) -> None:
        self.buffer.push(s, a, r, s2, done)

    def learn(self, nS: int) -> Optional[float]:
        if len(self.buffer) < self.batch_size:
            return None

        batch  = self.buffer.sample(self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        def onehot(idx_list):
            t = torch.zeros(len(idx_list), nS)
            for i, idx in enumerate(idx_list):
                t[i, idx] = 1.0
            return t

        S  = onehot(states)
        S2 = onehot(next_states)
        A  = torch.tensor(actions, dtype=torch.long).unsqueeze(1)
        R  = torch.tensor(rewards, dtype=torch.float32)
        D  = torch.tensor(dones,   dtype=torch.float32)

        # Current Q values
        Q_pred = self.online(S).gather(1, A).squeeze(1)

        # Target Q values (no gradient)
        with torch.no_grad():
            Q_next = self.target(S2).max(dim=1).values
            Q_tgt  = R + self.gamma * Q_next * (1.0 - D)

        loss = self.loss_fn(Q_pred, Q_tgt)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.online.parameters(), 10.0)
        self.optimizer.step()

        self._step += 1
        if self._step % self.target_update_freq == 0:
            self.target.load_state_dict(self.online.state_dict())

        return float(loss.item())
