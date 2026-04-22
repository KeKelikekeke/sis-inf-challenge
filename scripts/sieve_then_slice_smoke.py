"""Minimal smoke command for the Stage-4 Sieve-Then-Slice scaffold."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from sisinf.sieve_then_slice import run_sieve_then_slice_on_reduced_basis  # noqa: E402
from sisinf.two_step import DiagnosticReducedRowBackend  # noqa: E402
from sisinf.types import Instance  # noqa: E402


def _tiny_instance() -> Instance:
    return Instance(
        name="sieve_then_slice_smoke",
        index=0,
        n=2,
        m=2,
        q=7,
        gamma=7,
        A=np.array([[1, 2], [3, 4]], dtype=np.int64),
        t=np.zeros(2, dtype=np.int64),
        require_l2_ge_q=False,
        homogeneous=True,
        source_path=Path("memory"),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Stage-4 Sieve-Then-Slice scaffold smoke command.")
    parser.add_argument("--kappa", type=int, default=2, help="Base sieving dimension. Default: 2.")
    parser.add_argument("--target-size", type=int, default=5, help="Target list size. Default: 5.")
    parser.add_argument("--base-top-k", type=int, default=2, help="Base diagnostic backend top_k. Default: 2.")
    parser.add_argument("--upper-top-k", type=int, default=2, help="Upper diagnostic backend top_k. Default: 2.")
    parser.add_argument("--verbose", action="store_true", help="Enable INFO logs.")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING, format="[%(levelname)s] %(name)s: %(message)s")

    inst = _tiny_instance()
    reduced_basis = np.array(
        [
            [-1, -3, 1, 0],
            [-2, -4, 0, 1],
            [7, 0, 0, 0],
            [0, 7, 0, 0],
        ],
        dtype=np.int64,
    )
    result = run_sieve_then_slice_on_reduced_basis(
        inst,
        reduced_basis,
        kappa=args.kappa,
        target_size=args.target_size,
        base_backend=DiagnosticReducedRowBackend(top_k=args.base_top_k),
        upper_backend=DiagnosticReducedRowBackend(top_k=args.upper_top_k),
    )

    print(f"kappa: {result.kappa}")
    print(f"phi: {result.phi}")
    print(f"backend_diagnostic_only: {result.backend_diagnostic_only}")
    print(f"base_list_count: {result.base_list_count}")
    print(f"upper_list_count: {result.upper_list_count}")
    print(f"lifted_T_count: {result.lifted_t_count}")
    print(f"slicer_output_count: {result.slicer_output_count}")
    print(f"final_candidate_count: {result.final_candidate_count}")
    for idx, vec in enumerate(result.vectors[:10], start=1):
        print(f"vector[{idx}]: {vec.tolist()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
