from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sisinf.two_step import (  # noqa: E402
    DiagnosticReducedRowBackend,
    TwoStepReductionTarget,
    extract_projected_sublattice,
    run_two_step_on_reduced_basis,
    summarize_required_list_size,
)
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


class FakeBackend:
    name = "fake_backend"
    diagnostic_only = False

    def __init__(self) -> None:
        self.calls = 0

    def generate_short_vector_list(self, inst: Instance, projected) -> list[np.ndarray]:
        self.calls += 1
        assert projected.kappa == 2
        return [projected.basis_rows[0].copy()]


def test_two_step_interface_objects_are_constructible() -> None:
    target = TwoStepReductionTarget(beta=20, target_rhf=1.01, max_loops=2)
    assert target.beta == 20
    assert target.target_rhf == 1.01
    assert "beta=20" in target.describe()


def test_extract_projected_sublattice_uses_first_kappa_rows() -> None:
    projected = extract_projected_sublattice(_reduced_basis(), kappa=2)
    assert projected.kappa == 2
    assert projected.ambient_dimension == 4
    np.testing.assert_array_equal(projected.basis_rows, _reduced_basis()[:2, :])


def test_short_vector_backend_is_pluggable() -> None:
    backend = FakeBackend()
    result = run_two_step_on_reduced_basis(
        _inst(),
        _reduced_basis(),
        reduction_target=TwoStepReductionTarget(beta=2, max_loops=1),
        kappa=2,
        backend=backend,
    )
    assert backend.calls == 1
    assert result.backend_name == "fake_backend"
    assert result.backend_diagnostic_only is False
    assert len(result.vectors) == 1


def test_diagnostic_backend_logs_diagnostic_only(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.INFO)
    result = run_two_step_on_reduced_basis(
        _inst(),
        _reduced_basis(),
        reduction_target=TwoStepReductionTarget(beta=2, max_loops=1),
        kappa=2,
        backend=DiagnosticReducedRowBackend(top_k=2),
    )
    assert result.backend_diagnostic_only is True
    text = "\n".join(record.getMessage() for record in caplog.records)
    assert "diagnostic-only" in text
    assert "diagnostic_reduced_row_backend" in text


def test_integer_required_size_is_at_least_one() -> None:
    summary = summarize_required_list_size(p_success=0.1, p_single=0.2)
    assert summary.raw_required_size > 0.0
    assert summary.integer_required_size >= 1


def test_integer_required_size_handles_infinite_case() -> None:
    summary = summarize_required_list_size(p_success=0.5, p_single=0.0)
    assert summary.raw_required_size == float("inf")
    assert summary.integer_required_size == float("inf")
