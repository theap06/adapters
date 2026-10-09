"""
Select the stratified parity subset used for the repeated parity runs.

Tasks are stratified by venue-year (e.g. ``iclr2023``, ``neurips2024``) and
allocated proportionally (largest-remainder rounding), then sampled with a
fixed seed so the subset is reproducible.

Usage (from this directory, after parity_runner.py has cloned upstream):
    python select_parity_subset.py > parity_subset.txt
"""

from __future__ import annotations

import argparse
import random
from collections import defaultdict
from pathlib import Path

_ADAPTER_DIR = Path(__file__).resolve().parent
DEFAULT_TASKS_DIR = _ADAPTER_DIR / ".upstream_mlrbench" / "tasks"


def select(task_ids: list[str], size: int, seed: int) -> list[str]:
    strata: dict[str, list[str]] = defaultdict(list)
    for task_id in sorted(task_ids):
        strata[task_id.split("_", 1)[0]].append(task_id)

    quotas = {k: len(v) * size / len(task_ids) for k, v in strata.items()}
    alloc = {k: int(q) for k, q in quotas.items()}
    remaining = size - sum(alloc.values())
    for k in sorted(quotas, key=lambda k: (alloc[k] - quotas[k], k))[:remaining]:
        alloc[k] += 1

    rng = random.Random(seed)
    chosen: list[str] = []
    for k in sorted(strata):
        chosen.extend(rng.sample(strata[k], alloc[k]))
    return sorted(chosen)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tasks-dir", type=Path, default=DEFAULT_TASKS_DIR)
    p.add_argument("--size", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    task_ids = [f.stem for f in args.tasks_dir.glob("*.md")]
    if args.size > len(task_ids):
        raise SystemExit(f"--size {args.size} exceeds {len(task_ids)} tasks")
    for task_id in select(task_ids, args.size, args.seed):
        print(task_id)


if __name__ == "__main__":
    main()
