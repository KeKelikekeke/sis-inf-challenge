from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sisinf.hom_experiments import load_homogeneous_instance, run_hom_instance_workflow, scan_hom_parameter_grid, write_scan_csv  # noqa: E402
from sisinf.two_step import TwoStepReductionTarget  # noqa: E402


class FixedVectorBackend:
    def __init__(self, vectors: list[np.ndarray], name: str = "fixed_backend", diagnostic_only: bool = True) -> None:
        self.vectors = [np.asarray(vec, dtype=np.int64).reshape(-1).copy() for vec in vectors]
        self.name = name
        self.diagnostic_only = diagnostic_only

    def generate_short_vector_list(self, inst, projected) -> list[np.ndarray]:
        del inst, projected
        return [vec.copy() for vec in self.vectors]


def _write_custom_problem(tmp_path: Path) -> Path:
    path = tmp_path / "hom_problem.txt"
    path.write_text(
        "\n".join(
            [
                "A = [[1, 0], [0, 1]]",
                "q = 7",
                "gamma = 3",
            ]
        ),
        encoding="utf-8",
    )
    return path


def test_load_homogeneous_instance_defaults_missing_t_to_zero(tmp_path: Path) -> None:
    inst = load_homogeneous_instance(input_path=_write_custom_problem(tmp_path))
    assert inst.homogeneous is True
    assert inst.n == 2
    assert inst.m == 2
    np.testing.assert_array_equal(inst.t, np.zeros(2, dtype=np.int64))


def test_run_hom_instance_workflow_runs_on_small_instance(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr("sisinf.solver_restricted_hom.run_lll_on_row_basis", lambda B_row: np.asarray(B_row, dtype=np.int64))
    monkeypatch.setattr("sisinf.solver_restricted_hom.run_bkz_on_row_basis", lambda B_row, beta, max_loops=2: np.asarray(B_row, dtype=np.int64))

    inst = load_homogeneous_instance(input_path=_write_custom_problem(tmp_path))
    backend = FixedVectorBackend([np.array([0, 0, 1, 0], dtype=np.int64)])
    summary = run_hom_instance_workflow(
        inst,
        kappa=2,
        reduction_target=TwoStepReductionTarget(beta=2, target_rhf=1.01, max_loops=1),
        p_success=0.5,
        flexible_backend=backend,
        sieve_base_backend=backend,
        sieve_upper_backend=backend,
    )
    record = summary.to_record()
    assert record["selected_branch"] in {"flexible_d4f", "sieve_then_slice"}
    assert record["produced_list_size"] >= 0
    assert record["backend_diagnostic_only"] is True


def test_scan_hom_parameter_grid_produces_rows_and_csv(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr("sisinf.solver_restricted_hom.run_lll_on_row_basis", lambda B_row: np.asarray(B_row, dtype=np.int64))
    monkeypatch.setattr("sisinf.solver_restricted_hom.run_bkz_on_row_basis", lambda B_row, beta, max_loops=2: np.asarray(B_row, dtype=np.int64))

    inst = load_homogeneous_instance(input_path=_write_custom_problem(tmp_path))
    backend = FixedVectorBackend([np.array([0, 0, 1, 0], dtype=np.int64)])
    rows = scan_hom_parameter_grid(
        inst,
        kappas=[1, 2],
        target_rhfs=[1.01],
        p_successes=[0.5, 0.8],
        beta=2,
        max_loops=1,
        flexible_backend=backend,
        sieve_base_backend=backend,
        sieve_upper_backend=backend,
    )
    assert len(rows) == 4
    assert {"selected_branch", "produced_list_size", "restriction_pass_count", "backend_diagnostic_only"} <= set(rows[0])

    csv_path = tmp_path / "scan.csv"
    write_scan_csv(rows, csv_path)
    with csv_path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        loaded = list(reader)
    assert len(loaded) == len(rows)
