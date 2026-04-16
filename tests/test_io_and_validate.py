from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sisinf.io import _normalize_A_from_column_format, parse_problem_file
from sisinf.types import Instance
from sisinf.validate import validate_candidate


def _patch_spec(monkeypatch: pytest.MonkeyPatch, spec: dict[str, Any]) -> None:
    """Patch io.get_problem_spec for small temporary test instances."""

    monkeypatch.setattr("sisinf.io.get_problem_spec", lambda index: dict(spec))


def _write_problem(tmp_path: Path, text: str) -> Path:
    """Write a temporary problem text file."""

    path = tmp_path / "problem.txt"
    path.write_text(text, encoding="utf-8")
    return path


def test_A_column_format_is_transposed() -> None:
    """Raw column format shape (m, n) is normalized to (n, m)."""

    A = _normalize_A_from_column_format([[1, 2], [3, 4], [5, 6]], n=2, m=3)
    np.testing.assert_array_equal(A, np.array([[1, 3, 5], [2, 4, 6]], dtype=np.int64))


def test_homogeneous_missing_t_is_zero_filled(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Homogeneous instances may omit t and are normalized to zero target."""

    _patch_spec(monkeypatch, {"n": 2, "m": 3, "q": 7, "gamma": 3, "homogeneous": True, "require_l2_ge_q": False})
    inst = parse_problem_file(_write_problem(tmp_path, "A = [[1, 2], [3, 4], [5, 6]]\n"), index=1)
    np.testing.assert_array_equal(inst.t, np.zeros(2, dtype=np.int64))


def test_inhomogeneous_missing_t_raises(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Inhomogeneous instances must provide t."""

    _patch_spec(monkeypatch, {"n": 2, "m": 3, "q": 7, "gamma": 3, "homogeneous": False, "require_l2_ge_q": False})
    with pytest.raises(ValueError, match="Missing t"):
        parse_problem_file(_write_problem(tmp_path, "A = [[1, 2], [3, 4], [5, 6]]\n"), index=2)


def test_dimension_mismatch_raises(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Unexpected raw A dimensions produce a clear error."""

    _patch_spec(monkeypatch, {"n": 2, "m": 3, "q": 7, "gamma": 3, "homogeneous": True, "require_l2_ge_q": False})
    with pytest.raises(ValueError, match=r"expected column-format shape \(3, 2\)"):
        parse_problem_file(_write_problem(tmp_path, "A = [[1, 2], [3, 4]]\n"), index=1)


def test_valid_candidate_passes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A candidate satisfying congruence and infinity bounds passes."""

    _patch_spec(monkeypatch, {"n": 2, "m": 2, "q": 7, "gamma": 3, "homogeneous": False, "require_l2_ge_q": False})
    inst = parse_problem_file(_write_problem(tmp_path, '{"A": [[1, 0], [0, 1]], "t": [1, 2]}'), index=2)
    cand = validate_candidate(inst, u=np.array([0, 0]), v=np.array([1, 2]))
    assert cand.congruence_ok is True
    assert cand.valid_main is True


def test_candidate_exceeding_gamma_fails(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A candidate outside the infinity bound fails main validation."""

    _patch_spec(monkeypatch, {"n": 2, "m": 2, "q": 7, "gamma": 3, "homogeneous": False, "require_l2_ge_q": False})
    inst = parse_problem_file(_write_problem(tmp_path, "A = [[0, 0], [0, 0]]\nt = [4, 0]\n"), index=2)
    cand = validate_candidate(inst, u=np.array([4, 0]), v=np.array([0, 0]))
    assert cand.congruence_ok is True
    assert cand.valid_main is False


def test_l2_extra_condition_triggers() -> None:
    """The extra l2 lower bound is enforced when requested."""

    inst = Instance("problem5", 5, 2, 2, 5, 5, np.zeros((2, 2), dtype=np.int64), np.zeros(2, dtype=np.int64), True, True, Path("memory"))
    assert validate_candidate(inst, np.array([1, 1]), np.array([0, 0])).valid_extra is False
    assert validate_candidate(inst, np.array([3, 4]), np.array([0, 0])).valid_extra is True
