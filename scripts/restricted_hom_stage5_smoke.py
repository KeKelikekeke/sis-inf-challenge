from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sisinf.solver_restricted_hom import run_restricted_svp_dispatcher_on_reduced_basis  # noqa: E402
from sisinf.two_step import DiagnosticReducedRowBackend, TwoStepReductionTarget  # noqa: E402
from sisinf.types import Instance  # noqa: E402


def main() -> None:
    inst = Instance(
        name="stage5_smoke",
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
    B_row_red = np.array(
        [
            [-10, -10, 0, 10],
            [-10, 0, 10, 0],
            [20, 0, 0, 0],
            [0, 20, 0, 0],
        ],
        dtype=np.int64,
    )

    result = run_restricted_svp_dispatcher_on_reduced_basis(
        inst,
        B_row_red,
        reduction_target=TwoStepReductionTarget(beta=2, target_rhf=1.01, max_loops=1),
        kappa=2,
        p_success=0.5,
        flexible_backend=DiagnosticReducedRowBackend(top_k=4),
        sieve_base_backend=DiagnosticReducedRowBackend(top_k=4),
        sieve_upper_backend=DiagnosticReducedRowBackend(top_k=4),
    )

    print(f"len_bound={result.len_bound}")
    print(f"P(len)={result.single_vector_pass_probability}")
    print(f"raw_required_size={result.raw_required_size}")
    print(f"integer_required_size={result.integer_required_size}")
    print(f"threshold_size={result.threshold_size}")
    print(f"selected_branch={result.selected_branch}")
    print(f"produced_list_size={result.candidate_count_before_restriction}")
    print(f"restriction_pass_count={result.restriction_pass_count}")
    print(f"backend_diagnostic_only={result.backend_diagnostic_only}")


if __name__ == "__main__":
    main()
