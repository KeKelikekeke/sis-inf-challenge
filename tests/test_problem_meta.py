from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sisinf.problem_meta import PROBLEM_SPECS, get_problem_spec, is_homogeneous_problem


def test_all_problem_specs_are_correct() -> None:
    """The ten challenge problem specs are centrally recorded."""

    expected = {
        1: (100, 100, 100, 15, True, False),
        2: (100, 100, 100, 15, False, False),
        3: (120, 120, 120, 16, True, False),
        4: (120, 120, 120, 16, False, False),
        5: (120, 120, 120, 16, True, True),
        6: (140, 140, 140, 17, True, False),
        7: (140, 140, 140, 17, False, False),
        8: (140, 140, 140, 17, True, True),
        9: (160, 160, 160, 18, True, False),
        10: (160, 160, 160, 18, False, False),
    }
    assert set(PROBLEM_SPECS) == set(expected)
    for index, (n, m, q, gamma, homogeneous, require_l2_ge_q) in expected.items():
        spec = get_problem_spec(index)
        assert (spec["n"], spec["m"], spec["q"], spec["gamma"]) == (n, m, q, gamma)
        assert spec["homogeneous"] is homogeneous
        assert spec["require_l2_ge_q"] is require_l2_ge_q


def test_homogeneous_problem_indices() -> None:
    """Homogeneous and inhomogeneous problem groups match the statement."""

    for index in (1, 3, 5, 6, 8, 9):
        assert is_homogeneous_problem(index) is True
    for index in (2, 4, 7, 10):
        assert is_homogeneous_problem(index) is False


def test_l2_extra_condition_indices() -> None:
    """Only problems 5 and 8 require the extra lower l2 condition."""

    required = {idx for idx, spec in PROBLEM_SPECS.items() if spec["require_l2_ge_q"]}
    assert required == {5, 8}
