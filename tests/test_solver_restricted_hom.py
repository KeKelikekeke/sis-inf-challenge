from __future__ import annotations

import logging
import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sisinf.restricted_svp import RestrictedSVPProblem  # noqa: E402
from sisinf.solver_restricted_hom import run_restricted_svp_dispatcher_on_reduced_basis, solve_homogeneous_restricted_svp  # noqa: E402
from sisinf.two_step import DiagnosticReducedRowBackend, TwoStepReductionTarget  # noqa: E402
from sisinf.types import Candidate, Instance  # noqa: E402


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
            [-1, -1, 0, 1],
            [-1, 0, 1, 0],
            [2, 0, 0, 0],
            [0, 2, 0, 0],
        ],
        dtype=np.int64,
    )


class FixedVectorBackend:
    def __init__(self, vectors: list[np.ndarray], name: str = "fixed_backend", diagnostic_only: bool = False) -> None:
        self._vectors = [np.asarray(vec, dtype=np.int64).reshape(-1).copy() for vec in vectors]
        self.name = name
        self.diagnostic_only = diagnostic_only

    def generate_short_vector_list(self, inst: Instance, projected) -> list[np.ndarray]:
        del inst, projected
        return [vec.copy() for vec in self._vectors]


def _restricted_problem(
    predicate,
    probability_fn,
) -> RestrictedSVPProblem:
    return RestrictedSVPProblem(
        name="test_restricted_problem",
        dimension=4,
        bound=7,
        predicate=predicate,
        probability_fn=probability_fn,
        source_instance=_inst(),
        heuristic_notes=("test-only restricted problem",),
    )


def test_integer_required_size_is_at_least_one_in_stage5_result() -> None:
    result = run_restricted_svp_dispatcher_on_reduced_basis(
        _inst(),
        _reduced_basis(),
        reduction_target=TwoStepReductionTarget(beta=2, target_rhf=1.01, max_loops=1),
        kappa=2,
        p_success=0.1,
        restricted_problem=_restricted_problem(
            predicate=lambda cand: True,
            probability_fn=lambda len_bound: 0.2,
        ),
        flexible_backend=FixedVectorBackend([np.array([0, 0, 0, 1], dtype=np.int64)]),
    )
    assert result.raw_required_size > 0.0
    assert result.integer_required_size >= 1


def test_dispatcher_selects_flexible_d4f_on_small_required_size() -> None:
    result = run_restricted_svp_dispatcher_on_reduced_basis(
        _inst(),
        _reduced_basis(),
        reduction_target=TwoStepReductionTarget(beta=2, target_rhf=1.01, max_loops=1),
        kappa=2,
        p_success=0.5,
        restricted_problem=_restricted_problem(
            predicate=lambda cand: True,
            probability_fn=lambda len_bound: 0.9,
        ),
        flexible_backend=FixedVectorBackend([np.array([0, 0, 0, 1], dtype=np.int64)]),
    )
    assert result.integer_required_size <= result.threshold_size
    assert result.selected_branch == "flexible_d4f"


def test_dispatcher_selects_sieve_then_slice_on_large_required_size() -> None:
    result = run_restricted_svp_dispatcher_on_reduced_basis(
        _inst(),
        _reduced_basis(),
        reduction_target=TwoStepReductionTarget(beta=2, target_rhf=1.01, max_loops=1),
        kappa=2,
        p_success=0.5,
        restricted_problem=_restricted_problem(
            predicate=lambda cand: True,
            probability_fn=lambda len_bound: 0.01,
        ),
        sieve_base_backend=FixedVectorBackend([np.array([0, 0, 0, 1], dtype=np.int64)], name="base_backend"),
        sieve_upper_backend=FixedVectorBackend([np.array([1, 0, 0, 0], dtype=np.int64)], name="upper_backend"),
    )
    assert result.integer_required_size > result.threshold_size
    assert result.selected_branch == "sieve_then_slice"


def test_restriction_predicate_is_only_applied_in_final_scan() -> None:
    predicate_calls: list[Candidate] = []

    def predicate(cand: Candidate) -> bool:
        predicate_calls.append(cand)
        return cand.linf_u <= 1 and cand.linf_v <= 1

    result = run_restricted_svp_dispatcher_on_reduced_basis(
        _inst(),
        _reduced_basis(),
        reduction_target=TwoStepReductionTarget(beta=2, target_rhf=1.01, max_loops=1),
        kappa=2,
        p_success=0.5,
        restricted_problem=_restricted_problem(
            predicate=predicate,
            probability_fn=lambda len_bound: 0.01,
        ),
        sieve_base_backend=FixedVectorBackend(
            [
                np.array([0, 0, 0, 1], dtype=np.int64),
                np.array([3, 0, 0, 1], dtype=np.int64),
            ],
            name="base_backend",
        ),
        sieve_upper_backend=FixedVectorBackend([np.array([1, 0, 0, 0], dtype=np.int64)], name="upper_backend"),
    )
    assert result.selected_branch == "sieve_then_slice"
    assert result.candidate_count_before_restriction >= 2
    assert 0 < result.candidate_count_after_restriction < result.candidate_count_before_restriction
    assert len(predicate_calls) == result.candidate_count_before_restriction


def test_diagnostic_backend_limitation_stays_explicit(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.INFO)
    result = run_restricted_svp_dispatcher_on_reduced_basis(
        _inst(),
        _reduced_basis(),
        reduction_target=TwoStepReductionTarget(beta=2, target_rhf=1.01, max_loops=1),
        kappa=2,
        p_success=0.5,
        restricted_problem=_restricted_problem(
            predicate=lambda cand: True,
            probability_fn=lambda len_bound: 0.9,
        ),
        flexible_backend=DiagnosticReducedRowBackend(top_k=2),
    )
    text = "\n".join(record.getMessage() for record in caplog.records)
    assert result.backend_diagnostic_only is True
    assert "backend_diagnostic_only=True" in text
    assert any("diagnostic-only backend limitation" in item for item in result.limitations)


def test_homogeneous_stage5_entry_returns_structured_result(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("sisinf.solver_restricted_hom.run_lll_on_row_basis", lambda B_row: np.asarray(B_row, dtype=np.int64))
    monkeypatch.setattr("sisinf.solver_restricted_hom.run_bkz_on_row_basis", lambda B_row, beta, max_loops=2: np.asarray(B_row, dtype=np.int64))

    result = solve_homogeneous_restricted_svp(
        _inst(),
        reduction_target=TwoStepReductionTarget(beta=2, target_rhf=1.01, max_loops=1),
        kappa=2,
        p_success=0.5,
        restricted_problem=_restricted_problem(
            predicate=lambda cand: cand.linf_v <= 1,
            probability_fn=lambda len_bound: 0.9,
        ),
        flexible_backend=FixedVectorBackend([np.array([0, 0, 0, 1], dtype=np.int64)]),
    )

    assert result.selected_branch == "flexible_d4f"
    assert result.candidate_count_after_restriction == 1
    assert math.isfinite(result.len_bound)
    assert isinstance(result.notes, tuple)
