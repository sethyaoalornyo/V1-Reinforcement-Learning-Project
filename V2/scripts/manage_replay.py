"""
scripts/manage_replay.py
========================
Utility for managing replay buffer storage across training runs.
Enforces a total-size cap and replaces stale data with fresh data.

Usage
-----
    # Rotate fresh → stale and cap at 50k total transitions
    python scripts/manage_replay.py rotate \
        --fresh replay_buffer/ddos/dqn/fresh/buffer.json \
        --stale replay_buffer/ddos/dqn/stale/buffer.json \
        --max_total 50000

    # Show buffer stats
    python scripts/manage_replay.py stats \
        --path replay_buffer/ddos/dqn/fresh/buffer.json

    # Clear a buffer
    python scripts/manage_replay.py clear \
        --path replay_buffer/ddos/dqn/stale/buffer.json

Design rationale
----------------
Replay buffers can grow unbounded and fill disk.  This script:
  1. Merges stale + fresh into a single combined buffer.
  2. Enforces `max_total` by keeping only the most recent entries
     (the circular deque evicts oldest automatically).
  3. Saves the combined buffer back to `stale_path` so future runs
     can continue building on previous experience.
  4. Clears the `fresh_path` after rotation so new training starts
     writing fresh experiences into an empty buffer.

Cloud storage note: swap the save() / load() calls with cloud SDK
uploads/downloads to enable cloud-backed replay buffers.
"""

from __future__ import annotations
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.replay_buffer import ReplayBuffer, rotate_replay


# ------------------------------------------------------------------ #
#  Sub-commands                                                         #
# ------------------------------------------------------------------ #

def cmd_rotate(args) -> None:
    """Merge fresh into stale and enforce max_total cap."""
    os.makedirs(os.path.dirname(args.stale) if os.path.dirname(args.stale) else ".", exist_ok=True)

    rotate_replay(
        fresh_path=args.fresh,
        stale_path=args.stale,
        max_total=args.max_total,
        seed=args.seed,
    )

    # Clear the fresh buffer so next training run starts clean
    if args.clear_fresh:
        empty = ReplayBuffer(capacity=args.max_total)
        empty.save(args.fresh)
        print(f"[manage_replay] Fresh buffer cleared → {args.fresh}")


def cmd_stats(args) -> None:
    """Print statistics about an existing buffer."""
    if not os.path.exists(args.path):
        print(f"[manage_replay] File not found: {args.path}")
        return
    buf = ReplayBuffer()
    buf.load(args.path)
    stats = buf.stats()
    print(f"\nBuffer stats for: {args.path}")
    for k, v in stats.items():
        print(f"  {k:<15}: {v}")
    print()

    # Optional breakdown by terminal vs non-terminal
    n_done = sum(1 for e in buf._buffer if e.done)
    print(f"  Terminal (done=True) : {n_done} ({100*n_done/max(len(buf),1):.1f}%)")
    print(f"  Non-terminal         : {len(buf) - n_done}")


def cmd_clear(args) -> None:
    """Clear (empty) a buffer file without deleting it."""
    assert args.path, "--path is required for 'clear'"
    buf = ReplayBuffer()
    buf.save(args.path)
    print(f"[manage_replay] Buffer cleared → {args.path}")


def cmd_list(args) -> None:
    """List all buffer files under the replay_buffer/ directory."""
    root = args.root
    if not os.path.isdir(root):
        print(f"[manage_replay] Directory not found: {root}")
        return
    print(f"\nReplay buffers under {root}/")
    total_exp = 0
    for dirpath, _, filenames in os.walk(root):
        for fname in filenames:
            if fname.endswith(".json"):
                fpath = os.path.join(dirpath, fname)
                try:
                    with open(fpath) as f:
                        data = json.load(f)
                    n = data.get("size", 0)
                    cap = data.get("capacity", "?")
                    print(f"  {fpath:<55}  {n:>6} / {cap} experiences")
                    total_exp += n
                except Exception:
                    print(f"  {fpath:<55}  [unreadable]")
    print(f"\n  Total experiences across all buffers: {total_exp}\n")


# ------------------------------------------------------------------ #
#  Entry point                                                          #
# ------------------------------------------------------------------ #

def main():
    parser = argparse.ArgumentParser(
        description="Replay Buffer Management Utility",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # rotate
    p_rot = sub.add_parser("rotate", help="Merge fresh → stale and enforce size cap")
    p_rot.add_argument("--fresh",      required=True, help="Path to fresh buffer JSON")
    p_rot.add_argument("--stale",      required=True, help="Path to stale buffer JSON")
    p_rot.add_argument("--max_total",  type=int, default=50_000, help="Maximum combined experiences")
    p_rot.add_argument("--seed",       type=int, default=0)
    p_rot.add_argument("--clear_fresh", action="store_true", default=True,
                       help="Clear fresh buffer after rotation (default: True)")

    # stats
    p_stats = sub.add_parser("stats", help="Print statistics about a buffer")
    p_stats.add_argument("--path", required=True, help="Path to buffer JSON")

    # clear
    p_clear = sub.add_parser("clear", help="Empty a buffer file")
    p_clear.add_argument("--path", required=True)

    # list
    p_list = sub.add_parser("list", help="List all buffer files")
    p_list.add_argument("--root", default="replay_buffer", help="Root directory to search")

    args = parser.parse_args()

    if args.command == "rotate":
        cmd_rotate(args)
    elif args.command == "stats":
        cmd_stats(args)
    elif args.command == "clear":
        cmd_clear(args)
    elif args.command == "list":
        cmd_list(args)


if __name__ == "__main__":
    main()
