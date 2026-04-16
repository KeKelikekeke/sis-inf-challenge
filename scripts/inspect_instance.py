"""Inspect a challenge instance and optionally validate a manual candidate."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from sisinf.io import load_problem  # noqa: E402
from sisinf.validate import format_candidate_summary, validate_candidate  # noqa: E402


def _load_json_vector(path: Path) -> np.ndarray:
    """Load a one-dimensional integer vector from a JSON array file."""

    data = json.loads(path.read_text(encoding="utf-8"))
    return np.asarray(data, dtype=np.int64).reshape(-1)


def main(argv: list[str] | None = None) -> int:
    """Run the instance inspection CLI."""

    parser = argparse.ArgumentParser(description="Inspect SIS infinity-norm challenge instances.")
    parser.add_argument("--problem", type=int, required=True, help="Problem index in 1..10.")
    parser.add_argument("--u-file", type=Path, help="JSON array file for candidate u.")
    parser.add_argument("--v-file", type=Path, help="JSON array file for candidate v.")
    args = parser.parse_args(argv)
    if (args.u_file is None) != (args.v_file is None):
        parser.error("--u-file and --v-file must be provided together")

    inst = load_problem(args.problem)
    print(f"problem: {inst.index} ({inst.name})")
    print(f"n: {inst.n}")
    print(f"m: {inst.m}")
    print(f"q: {inst.q}")
    print(f"gamma: {inst.gamma}")
    print(f"type: {'homogeneous' if inst.homogeneous else 'inhomogeneous'}")
    print(f"require_l2_ge_q: {inst.require_l2_ge_q}")
    print(f"A.shape: {inst.A.shape}")
    print(f"t_is_zero: {bool(inst.t is not None and np.all(inst.t == 0))}")
    print(f"source_path: {inst.source_path}")

    if args.u_file is None:
        return 0

    cand = validate_candidate(inst, _load_json_vector(args.u_file), _load_json_vector(args.v_file))
    print()
    print(format_candidate_summary(cand))
    return 0 if cand.valid_main and cand.valid_extra else 1


if __name__ == "__main__":
    raise SystemExit(main())
