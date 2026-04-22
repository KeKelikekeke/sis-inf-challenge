from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sisinf.flexible_d4f import (  # noqa: E402
    extract_flexible_d4f_projected_sublattice,
    flexible_d4f_gamma_factor,
    run_flexible_d4f_on_reduced_basis,
)
from sisinf.two_step import DiagnosticReducedRowBackend, TwoStepReductionTarget, extract_projected_sublattice  # noqa: E402
from sisinf.types import Instance  # noqa: E402


def _inst() -> Instance:
    return Instance(
        name="small",
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


def _reduced_basis() -> np.ndarray:
    return np.array(
        [
            [-1, -3, 1, 0],
            [-2, -4, 0, 1],
            [7, 0, 0, 0],
            [0, 7, 0, 0],
        ],
        dtype=np.int64,
    )


def test_f_prime_zero_reasonably_degenerates_to_stage2_slice() -> None:
    stage2 = extract_projected_sublattice(_reduced_basis(), kappa=3)
    d4f = extract_flexible_d4f_projected_sublattice(_reduced_basis(), kappa=3, f_prime=0)
    np.testing.assert_array_equal(stage2.basis_rows, d4f.basis_rows)


def test_projected_sublattice_shifts_from_l_0_kappa_to_l_fprime_kappa() -> None:
    projected = extract_flexible_d4f_projected_sublattice(_reduced_basis(), kappa=3, f_prime=1)
    np.testing.assert_array_equal(projected.basis_rows, _reduced_basis()[1:3, :])


def test_gamma_factor_decreases_as_f_prime_increases() -> None:
    values = [flexible_d4f_gamma_factor(target_rhf=1.01, ambient_dimension=4, f_prime=f_prime) for f_prime in [0, 1, 2]]
    assert values[0] > values[1] > values[2]


def test_diagnostic_backend_stays_explicit(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.INFO)
    result = run_flexible_d4f_on_reduced_basis(
        _inst(),
        _reduced_basis(),
        reduction_target=TwoStepReductionTarget(beta=2, target_rhf=1.01, max_loops=1),
        kappa=3,
        f_prime=1,
        backend=DiagnosticReducedRowBackend(top_k=3),
    )
    assert result.backend_diagnostic_only is True
    assert any("diagnostic-only" in note for note in result.notes)
    text = "\n".join(record.getMessage() for record in caplog.records)
    assert "diagnostic-only" in text


def test_length_threshold_filtering_is_applied() -> None:
    result = run_flexible_d4f_on_reduced_basis(
        _inst(),
        _reduced_basis(),
        reduction_target=TwoStepReductionTarget(beta=2, target_rhf=1.50, max_loops=1),
        kappa=3,
        f_prime=1,
        backend=DiagnosticReducedRowBackend(top_k=3),
    )
    assert result.candidate_count_before_lift == 2
    assert result.candidate_count_after_lift == 2
    assert result.candidate_count_after_length_filter < result.candidate_count_after_lift
