"""Central metadata for the ten challenge problem instances."""

from __future__ import annotations

from copy import deepcopy
from typing import Any


PROBLEM_SPECS: dict[int, dict[str, Any]] = {
    1: {"n": 100, "m": 100, "q": 100, "gamma": 15, "homogeneous": True, "require_l2_ge_q": False},
    2: {"n": 100, "m": 100, "q": 100, "gamma": 15, "homogeneous": False, "require_l2_ge_q": False},
    3: {"n": 120, "m": 120, "q": 120, "gamma": 16, "homogeneous": True, "require_l2_ge_q": False},
    4: {"n": 120, "m": 120, "q": 120, "gamma": 16, "homogeneous": False, "require_l2_ge_q": False},
    5: {"n": 120, "m": 120, "q": 120, "gamma": 16, "homogeneous": True, "require_l2_ge_q": True},
    6: {"n": 140, "m": 140, "q": 140, "gamma": 17, "homogeneous": True, "require_l2_ge_q": False},
    7: {"n": 140, "m": 140, "q": 140, "gamma": 17, "homogeneous": False, "require_l2_ge_q": False},
    8: {"n": 140, "m": 140, "q": 140, "gamma": 17, "homogeneous": True, "require_l2_ge_q": True},
    9: {"n": 160, "m": 160, "q": 160, "gamma": 18, "homogeneous": True, "require_l2_ge_q": False},
    10: {"n": 160, "m": 160, "q": 160, "gamma": 18, "homogeneous": False, "require_l2_ge_q": False},
}


def get_problem_spec(index: int) -> dict[str, Any]:
    """Return a copy of the metadata for one problem index."""

    if index not in PROBLEM_SPECS:
        known = ", ".join(str(i) for i in sorted(PROBLEM_SPECS))
        raise ValueError(f"Unknown problem index {index!r}; expected one of: {known}")
    return deepcopy(PROBLEM_SPECS[index])


def is_homogeneous_problem(index: int) -> bool:
    """Return whether the indexed problem is homogeneous."""

    return bool(get_problem_spec(index)["homogeneous"])
