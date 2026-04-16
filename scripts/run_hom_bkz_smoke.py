"""Run a minimal homogeneous SIS infinity-norm BKZ baseline smoke test."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from sisinf.io import load_problem  # noqa: E402
from sisinf.solver_hom_bkz import (  # noqa: E402
    solve_homogeneous_bkz_baseline,
    solve_homogeneous_bkz_with_search,
    summarize_candidate_list,
    summarize_candidate_pool,
)


def main(argv: list[str] | None = None) -> int:
    """Run the smoke CLI and return a process exit code."""

    parser = argparse.ArgumentParser(description="Run homogeneous SIS infinity-norm BKZ baseline smoke test.")
    parser.add_argument("--problem", type=int, default=1, help="Homogeneous problem index. Default: 1.")
    parser.add_argument("--beta", type=int, default=10, help="BKZ block size. Default: 10.")
    parser.add_argument("--max-loops", type=int, default=1, help="BKZ max loops. Default: 1.")
    parser.add_argument("--top-k", type=int, default=10, help="Number of reduced rows to validate. Default: 10.")
    parser.add_argument("--use-search", action="store_true", help="Expand reduced rows with pairwise candidate search.")
    parser.add_argument("--pair-max-base", type=int, default=20, help="Maximum base rows used for pairwise search.")
    parser.add_argument("--pair-budget", type=int, default=200, help="Maximum generated pairwise combination vectors.")
    args = parser.parse_args(argv)

    inst = load_problem(args.problem)
    if not inst.homogeneous:
        raise ValueError(f"problem{args.problem} is inhomogeneous; smoke script only supports homogeneous instances")

    print(f"problem: {inst.index} ({inst.name})")
    print(f"n: {inst.n}")
    print(f"m: {inst.m}")
    print(f"q: {inst.q}")
    print(f"gamma: {inst.gamma}")
    print(f"beta: {args.beta}")
    print(f"max_loops: {args.max_loops}")
    print(f"top_k: {args.top_k}")
    print(f"use_search: {args.use_search}")
    if args.use_search:
        print(f"pair_max_base: {args.pair_max_base}")
        print(f"pair_budget: {args.pair_budget}")
    print()

    if args.use_search:
        cands = solve_homogeneous_bkz_with_search(
            inst,
            beta=args.beta,
            max_loops=args.max_loops,
            top_k=args.top_k,
            pair_max_base=args.pair_max_base,
            pair_budget=args.pair_budget,
        )
        print(summarize_candidate_pool(cands))
    else:
        cands = solve_homogeneous_bkz_baseline(
            inst,
            beta=args.beta,
            max_loops=args.max_loops,
            top_k=args.top_k,
        )
        print(summarize_candidate_list(cands))
    print()
    if any(cand.valid_main for cand in cands):
        print("found valid_main candidate")
    else:
        print("no valid_main candidate found")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
