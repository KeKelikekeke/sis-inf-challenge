from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sisinf.sieve_then_slice import (  # noqa: E402
    compute_sieve_then_slice_phi,
    extract_sieve_then_slice_projected_sublattice,
    lift_upper_sublattice_vector_identity_scaffold,
    modified_randomized_slicer_scaffold,
    run_sieve_then_slice_on_reduced_basis,
)
from sisinf.two_step import DiagnosticReducedRowBackend  # noqa: E402
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


def test_phi_is_nonnegative_and_nondecreasing_in_target_size() -> None:
    values = [compute_sieve_then_slice_phi(target_size=s, kappa=2) for s in [1, 2, 5, 10]]
    assert all(value >= 0 for value in values)
    assert values == sorted(values)


def test_upper_projected_sublattice_is_l_kappa_kappa_plus_phi() -> None:
    projected = extract_sieve_then_slice_projected_sublattice(_reduced_basis(), start=2, stop=4)
    np.testing.assert_array_equal(projected.basis_rows, _reduced_basis()[2:4, :])


def test_lift_interface_exists_and_is_observable() -> None:
    upper = extract_sieve_then_slice_projected_sublattice(_reduced_basis(), start=2, stop=4)
    vec = np.array([7, 0, 0, 0], dtype=np.int64)
    lifted = lift_upper_sublattice_vector_identity_scaffold(vec, upper)
    np.testing.assert_array_equal(lifted, vec)


def test_modified_randomized_slicer_scaffold_is_callable() -> None:
    T = [np.array([7, 0, 0, 0], dtype=np.int64)]
    Lsieve = [np.array([-1, -3, 1, 0], dtype=np.int64)]
    out = modified_randomized_slicer_scaffold(T, Lsieve, target_size=3)
    assert len(out) == 1
    np.testing.assert_array_equal(out[0], np.array([8, 3, -1, 0], dtype=np.int64))


def test_diagnostic_backend_stays_explicit(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.INFO)
    result = run_sieve_then_slice_on_reduced_basis(
        _inst(),
        _reduced_basis(),
        kappa=2,
        target_size=5,
        base_backend=DiagnosticReducedRowBackend(top_k=2),
        upper_backend=DiagnosticReducedRowBackend(top_k=2),
    )
    assert result.backend_diagnostic_only is True
    assert any("diagnostic-only" in note for note in result.notes)
    text = "\n".join(record.getMessage() for record in caplog.records)
    assert "diagnostic-only" in text
