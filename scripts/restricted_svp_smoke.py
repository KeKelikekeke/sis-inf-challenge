"""Minimal Stage-1 smoke command for restricted-SVP modeling."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from sisinf.probability import prob_infinity_norm_pass, required_list_size  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    """Run the smoke CLI."""

    parser = argparse.ArgumentParser(description="Stage-1 restricted-SVP smoke command for homogeneous SIS∞ modeling.")
    parser.add_argument("--gamma", type=float, required=True, help="Infinity-norm bound B (gamma).")
    parser.add_argument("--dim", type=int, required=True, help="Vector dimension d.")
    parser.add_argument("--len", dest="len_bound", type=float, required=True, help="Euclidean length bound len.")
    parser.add_argument("--p-success", type=float, required=True, help="Target success probability in [0, 1).")
    args = parser.parse_args(argv)

    p_len = prob_infinity_norm_pass(dim=args.dim, bound=args.gamma, len_bound=args.len_bound)
    size = required_list_size(p_success=args.p_success, p_single=p_len)

    print(f"gamma: {args.gamma}")
    print(f"dim: {args.dim}")
    print(f"len: {args.len_bound}")
    print(f"p_success: {args.p_success}")
    print(f"P(len): {p_len:.12f}")
    print(f"required_size: {size:.12f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
