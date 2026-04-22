"""Minimal smoke command for the Stage-3 FlexibleD4F scaffold."""

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

from sisinf.flexible_d4f import run_flexible_d4f_on_reduced_basis  # noqa: E402
from sisinf.two_step import DiagnosticReducedRowBackend, TwoStepReductionTarget  # noqa: E402
from sisinf.types import Instance  # noqa: E402


def _tiny_instance() -> Instance:
    return Instance(
        name="flexible_d4f_smoke",
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
    parser = argparse.ArgumentParser(description="Stage-3 FlexibleD4F scaffold smoke command.")
    parser.add_argument("--kappa", type=int, default=3, help="Projected sublattice end index. Default: 3.")
    parser.add_argument("--f-prime", type=int, default=1, help="FlexibleD4F free-dimension count. Default: 1.")
    parser.add_argument("--target-rhf", type=float, default=1.01, help="Explicit RHF / delta value. Default: 1.01.")
    parser.add_argument("--top-k", type=int, default=3, help="Diagnostic backend output cap. Default: 3.")
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
    result = run_flexible_d4f_on_reduced_basis(
        inst,
        reduced_basis,
        reduction_target=TwoStepReductionTarget(beta=2, target_rhf=args.target_rhf, max_loops=1),
        kappa=args.kappa,
        f_prime=args.f_prime,
        backend=DiagnosticReducedRowBackend(top_k=args.top_k),
    )

    print(f"backend: {result.backend_name}")
    print(f"diagnostic_only: {result.backend_diagnostic_only}")
    print(f"kappa: {result.projected_sublattice.kappa}")
    print(f"f_prime: {result.projected_sublattice.f_prime}")
    print(f"gamma_factor: {result.gamma_factor}")
    print(f"length_threshold: {result.length_threshold}")
    print(f"candidate_count_before_lift: {result.candidate_count_before_lift}")
    print(f"candidate_count_after_lift: {result.candidate_count_after_lift}")
    print(f"candidate_count_after_length_filter: {result.candidate_count_after_length_filter}")
    print(f"vector_count: {len(result.vectors)}")
    for idx, vec in enumerate(result.vectors, start=1):
        print(f"vector[{idx}]: {vec.tolist()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
