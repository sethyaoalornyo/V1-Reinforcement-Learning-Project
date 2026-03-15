from __future__ import annotations
import argparse
import json
import os
import random
from typing import List

import matplotlib.pyplot as plt

from mdp import build_ddos_style_mdp, save_mdp
from dp import policy_iteration, value_iteration


def save_artifacts(policy: List[int], V, Q, mdp, out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)

    save_mdp(mdp, os.path.join(out_dir, "mdp.json"))

    pretty_policy = {
        mdp.state_names[s]: mdp.action_names[a] for s, a in enumerate(policy)
    }

    with open(os.path.join(out_dir, "policy.json"), "w", encoding="utf-8") as f:
        json.dump(pretty_policy, f, indent=2)

    with open(os.path.join(out_dir, "V.json"), "w", encoding="utf-8") as f:
        json.dump({mdp.state_names[s]: float(V[s]) for s in range(mdp.nS)}, f, indent=2)

    with open(os.path.join(out_dir, "Q.json"), "w", encoding="utf-8") as f:
        q_dump = {
            mdp.state_names[s]: {mdp.action_names[a]: float(Q[s][a]) for a in range(mdp.nA)}
            for s in range(mdp.nS)
        }
        json.dump(q_dump, f, indent=2)

    plot_state_values(V, mdp, out_dir)
    plot_q_values(Q, mdp, out_dir)
    plot_policy_diagram(policy, mdp, out_dir)
    plot_transition_diagram(policy, mdp, out_dir)

    sim_states, sim_actions, sim_rewards = simulate_agent(mdp, policy, start_state=0, max_steps=15, seed=7)
    plot_simulation(sim_states, sim_actions, sim_rewards, mdp, out_dir)


def plot_state_values(V, mdp, out_dir: str) -> None:
    plt.figure(figsize=(8, 5))
    plt.bar(mdp.state_names, V)
    plt.title("State Values Learned by the Agent")
    plt.xlabel("States")
    plt.ylabel("Value V(s)")
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "state_values.png"))
    plt.show()
    plt.close()


def plot_q_values(Q, mdp, out_dir: str) -> None:
    states = list(range(mdp.nS))
    width = 0.25

    plt.figure(figsize=(10, 5))
    for a in range(mdp.nA):
        q_vals = [Q[s][a] for s in states]
        x_positions = [s + (a - 1) * width for s in states]
        plt.bar(x_positions, q_vals, width=width, label=mdp.action_names[a])

    plt.title("Q-Values for Each Action in Every State")
    plt.xlabel("States")
    plt.ylabel("Q(s, a)")
    plt.xticks(states, mdp.state_names, rotation=20)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "q_values.png"))
    plt.show()
    plt.close()


def plot_policy_diagram(policy, mdp, out_dir: str) -> None:
    plt.figure(figsize=(10, 3))
    plt.axis("off")

    y = 0.5
    x_positions = list(range(mdp.nS))

    for i, s in enumerate(range(mdp.nS)):
        x = x_positions[i]
        plt.text(
            x, y,
            f"{mdp.state_names[s]}\n↓\n{mdp.action_names[policy[s]]}",
            ha="center",
            va="center",
            bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="black")
        )

        if i < mdp.nS - 1:
            plt.annotate(
                "",
                xy=(x_positions[i + 1] - 0.35, y),
                xytext=(x + 0.35, y),
                arrowprops=dict(arrowstyle="->", lw=1.5)
            )

    plt.title("Policy Diagram: Best Action in Each State")
    plt.xlim(-0.5, mdp.nS - 0.5)
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "policy_diagram.png"))
    plt.show()
    plt.close()


def plot_transition_diagram(policy, mdp, out_dir: str) -> None:
    plt.figure(figsize=(12, 6))
    plt.axis("off")

    positions = {
        0: (0, 0),
        1: (2.5, 1.5),
        2: (5, 1.5),
        3: (7.5, 0),
        4: (10, 0),
    }

    # Draw nodes
    for s in range(mdp.nS):
        x, y = positions[s]
        plt.text(
            x, y,
            f"{mdp.state_names[s]}\n[{mdp.action_names[policy[s]]}]",
            ha="center",
            va="center",
            bbox=dict(boxstyle="circle,pad=0.5", fc="white", ec="black")
        )

    # Draw most likely transition under policy
    for s in range(mdp.nS):
        a = policy[s]
        transitions = mdp.transitions(s, a)
        best_transition = max(transitions, key=lambda t: t[0])
        _, s2, _, _ = best_transition

        if s != s2:
            x1, y1 = positions[s]
            x2, y2 = positions[s2]
            plt.annotate(
                "",
                xy=(x2, y2),
                xytext=(x1, y1),
                arrowprops=dict(arrowstyle="->", lw=2)
            )
        else:
            x, y = positions[s]
            plt.text(x, y - 1.0, "self-loop", ha="center")

    plt.title("Network-Style State Transition Diagram Under Learned Policy")
    plt.xlim(-1, 11)
    plt.ylim(-2, 3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "transition_diagram.png"))
    plt.show()
    plt.close()


def simulate_agent(mdp, policy, start_state: int = 0, max_steps: int = 15, seed: int = 7):
    random.seed(seed)

    states = [start_state]
    actions = []
    rewards = []

    s = start_state
    for _ in range(max_steps):
        a = policy[s]
        actions.append(a)

        transitions = mdp.transitions(s, a)

        rnum = random.random()
        cumulative = 0.0
        chosen = transitions[-1]

        for t in transitions:
            prob, s2, r, done = t
            cumulative += prob
            if rnum <= cumulative:
                chosen = t
                break

        prob, s2, r, done = chosen
        rewards.append(r)
        states.append(s2)
        s = s2

        if done:
            break

    return states, actions, rewards


def plot_simulation(states, actions, rewards, mdp, out_dir: str) -> None:
    steps = list(range(len(states)))

    plt.figure(figsize=(10, 5))
    plt.plot(steps, states, marker="o")
    plt.title("Simulation of Agent Moving Through States")
    plt.xlabel("Time Step")
    plt.ylabel("State Index")
    plt.yticks(range(mdp.nS), mdp.state_names)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "simulation_states.png"))
    plt.show()
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.plot(range(len(rewards)), rewards, marker="o")
    plt.title("Rewards Received During Simulation")
    plt.xlabel("Time Step")
    plt.ylabel("Reward")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "simulation_rewards.png"))
    plt.show()
    plt.close()


def print_results(policy, V, Q, mdp):
    print("\n=== Results ===")
    for s in range(mdp.nS):
        sname = mdp.state_names[s]
        aname = mdp.action_names[policy[s]]
        q_dict = {mdp.action_names[a]: round(Q[s][a], 3) for a in range(mdp.nA)}
        print(f"{sname:>12} -> {aname:>10} | V={V[s]: .3f} | Q={q_dict}")


def main():
    parser = argparse.ArgumentParser(description="V1 Term Project: Tabular RL via Dynamic Programming")
    parser.add_argument("--gamma", type=float, default=0.95, help="Discount factor")
    parser.add_argument("--method", choices=["policy_iteration", "value_iteration"], default="policy_iteration")
    parser.add_argument("--theta", type=float, default=1e-8, help="Convergence tolerance")
    parser.add_argument("--out", type=str, default="artifacts", help="Output directory for artifacts")
    args = parser.parse_args()

    mdp = build_ddos_style_mdp(gamma=args.gamma)

    if args.method == "policy_iteration":
        policy, V, Q = policy_iteration(mdp, theta=args.theta)
    else:
        policy, V, Q = value_iteration(mdp, theta=args.theta)

    print_results(policy, V, Q, mdp)
    save_artifacts(policy, V, Q, mdp, args.out)

    print(f"\nSaved artifacts to: {args.out}")
    print("Saved diagrams:")
    print("- state_values.png")
    print("- q_values.png")
    print("- policy_diagram.png")
    print("- transition_diagram.png")
    print("- simulation_states.png")
    print("- simulation_rewards.png")


if __name__ == "__main__":
    main()