"""
Deep Q-Network (DQN) — Mnih et al., 2015
==========================================
DQN combines Q-learning with two key tricks that make deep learning
stable for RL:

1. Experience Replay   : store transitions in a buffer, sample random
                         mini-batches to break temporal correlations.
2. Target Network      : use a periodically-frozen copy of the network
                         to compute TD targets, preventing the target
                         from "chasing" the prediction.

Why DQN for the DDoS environment?
-----------------------------------
Our foundation environment has a DISCRETE action space (ALLOW / RATE_LIMIT
/ BLOCK), which is the setting DQN was designed for.  Even though the
state space is tiny (5 states), the DQN implementation here is general
and scales directly to higher-dimensional inputs (e.g., traffic feature
vectors, sequence encodings of packet logs).

Architecture choice: MLP with one-hot state encoding.
Exploration: ε-greedy with exponential decay.
"""

from __future__ import annotations
import copy
import os
import random
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim

from src.utils.env_wrapper import TabularEnv
from src.utils.replay_buffer import ReplayBuffer
from src.utils.logger import TrainingLogger
from src.deep.networks import MLP, DuelingMLP, state_to_tensor, batch_states_to_tensor, save_checkpoint


# ------------------------------------------------------------------ #
#  DQN Agent                                                            #
# ------------------------------------------------------------------ #

class DQNAgent:
    """
    DQN agent with experience replay and a periodically-updated target net.

    Parameters
    ----------
    env              : the TabularEnv to train on
    hidden_dim       : width of hidden layers in the MLP
    lr               : learning rate for Adam optimiser
    gamma            : discount factor
    epsilon_start    : initial exploration probability
    epsilon_end      : minimum exploration probability
    epsilon_decay    : multiplicative decay applied per episode
    buffer_capacity  : maximum replay buffer size
    batch_size       : number of transitions sampled per gradient step
    target_update    : every N episodes, copy online → target network
    device           : "cpu" or "cuda"
    dueling          : use DuelingMLP architecture instead of plain MLP
    """

    def __init__(
        self,
        env: TabularEnv,
        hidden_dim: int = 64,
        lr: float = 1e-3,
        gamma: float = 0.95,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay: float = 0.995,
        buffer_capacity: int = 10_000,
        batch_size: int = 64,
        target_update: int = 20,
        device: str = "cpu",
        dueling: bool = False,
    ) -> None:
        self.env         = env
        self.nS          = env.nS
        self.nA          = env.nA
        self.gamma       = gamma
        self.epsilon     = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size  = batch_size
        self.target_update = target_update
        self.device      = device

        # Networks
        NetCls = DuelingMLP if dueling else MLP
        self.online = NetCls(self.nS, hidden_dim, self.nA).to(device)
        self.target = copy.deepcopy(self.online).to(device)
        self.target.eval()

        self.optimiser = optim.Adam(self.online.parameters(), lr=lr)
        self.buffer    = ReplayBuffer(buffer_capacity)
        self._step     = 0

    # ---------------------------------------------------------------- #
    #  Action selection                                                  #
    # ---------------------------------------------------------------- #

    def select_action(self, state: int) -> int:
        """ε-greedy action selection."""
        if random.random() < self.epsilon:
            return random.randint(0, self.nA - 1)
        with torch.no_grad():
            x = state_to_tensor(state, self.nS, self.device)
            return int(self.online(x).argmax(dim=1).item())

    # ---------------------------------------------------------------- #
    #  Learning step                                                     #
    # ---------------------------------------------------------------- #

    def _learn(self) -> float:
        """
        Sample a mini-batch from replay and do one gradient step.

        Loss = MSE(online Q(s,a), target r + γ max_{a'} target_net(s'))
        Returns the scalar loss value.
        """
        states, actions, rewards, next_states, dones = self.buffer.sample_as_tensors(
            self.batch_size
        )

        # Convert to tensors
        s_batch  = batch_states_to_tensor(states,      self.nS, self.device)
        s2_batch = batch_states_to_tensor(next_states, self.nS, self.device)
        a_batch  = torch.tensor(actions,  dtype=torch.long,  device=self.device)
        r_batch  = torch.tensor(rewards,  dtype=torch.float, device=self.device)
        d_batch  = torch.tensor(dones,    dtype=torch.float, device=self.device)

        # Online Q-values for taken actions
        q_pred = self.online(s_batch).gather(1, a_batch.unsqueeze(1)).squeeze(1)

        # Target Q-values using FROZEN target network
        with torch.no_grad():
            q_next = self.target(s2_batch).max(dim=1).values
            q_target = r_batch + self.gamma * q_next * (1.0 - d_batch)

        loss = nn.functional.mse_loss(q_pred, q_target)

        self.optimiser.zero_grad()
        loss.backward()
        # Gradient clipping for stability
        nn.utils.clip_grad_norm_(self.online.parameters(), max_norm=1.0)
        self.optimiser.step()

        self._step += 1
        return loss.item()

    # ---------------------------------------------------------------- #
    #  Training loop                                                     #
    # ---------------------------------------------------------------- #

    def train(
        self,
        num_episodes: int = 500,
        log_every: int = 50,
        checkpoint_dir: str = "checkpoints/dqn/ddos/run1",
        logger: Optional[TrainingLogger] = None,
    ) -> List[float]:
        """
        Full training loop.

        Returns episode returns for plotting.
        """
        ep_returns = []

        for ep in range(num_episodes):
            s    = self.env.reset()
            done = False
            G    = 0.0
            step = 0
            ep_loss = []

            while not done:
                a               = self.select_action(s)
                s2, r, done, _  = self.env.step(a)
                self.buffer.push(s, a, r, s2, done)
                G    += (self.gamma ** step) * r
                step += 1
                s    = s2

                # Learn once buffer is warm
                if self.buffer.is_ready(self.batch_size):
                    loss = self._learn()
                    ep_loss.append(loss)

            # Decay exploration
            self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

            # Periodically sync target network
            if (ep + 1) % self.target_update == 0:
                self.target.load_state_dict(self.online.state_dict())

            ep_returns.append(G)

            if logger:
                mean_loss = sum(ep_loss) / len(ep_loss) if ep_loss else 0.0
                logger.log_episode(ep, G, step, {"loss": mean_loss, "epsilon": self.epsilon})

            if (ep + 1) % log_every == 0:
                recent = sum(ep_returns[-log_every:]) / log_every
                print(
                    f"  Ep {ep+1:>5}/{num_episodes}  "
                    f"AvgReturn={recent:.3f}  ε={self.epsilon:.3f}"
                )

        # Save final checkpoints
        save_checkpoint(
            self.online, os.path.join(checkpoint_dir, "online.pt"),
            meta={"episodes": num_episodes, "epsilon": self.epsilon, "step": self._step},
        )
        save_checkpoint(self.target, os.path.join(checkpoint_dir, "target.pt"))
        self.buffer.save(f"replay_buffer/ddos/dqn/fresh/buffer.json")

        return ep_returns

    # ---------------------------------------------------------------- #
    #  Evaluation                                                        #
    # ---------------------------------------------------------------- #

    def evaluate(self, num_episodes: int = 50, seed: int = 99) -> float:
        """
        Run the greedy policy (ε=0) and return mean episode return.
        Does NOT modify the replay buffer or update any weights.
        """
        self.env.seed(seed)
        total = 0.0
        for _ in range(num_episodes):
            s, done, G, step = self.env.reset(), False, 0.0, 0
            while not done:
                with torch.no_grad():
                    x = state_to_tensor(s, self.nS, self.device)
                    a = int(self.online(x).argmax(dim=1).item())
                s, r, done, _ = self.env.step(a)
                G += (self.gamma ** step) * r
                step += 1
            total += G
        mean_return = total / num_episodes
        print(f"[DQN Eval] Mean return over {num_episodes} eps: {mean_return:.4f}")
        return mean_return

    def get_q_table(self) -> List[List[float]]:
        """
        Extract Q-values for all states (useful for comparison with
        tabular methods).  Returns a (nS × nA) list of lists.
        """
        q_table = []
        with torch.no_grad():
            for s in range(self.nS):
                x  = state_to_tensor(s, self.nS, self.device)
                qs = self.online(x).squeeze(0).tolist()
                q_table.append(qs)
        return q_table

    def get_policy(self) -> List[int]:
        """Return the greedy policy as a list of actions (one per state)."""
        q_table = self.get_q_table()
        return [max(range(self.nA), key=lambda a: q_table[s][a]) for s in range(self.nS)]
