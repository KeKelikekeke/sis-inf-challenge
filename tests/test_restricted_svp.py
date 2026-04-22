from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sisinf.probability import prob_infinity_norm_pass, required_list_size  # noqa: E402
from sisinf.restricted_svp import (  # noqa: E402
    make_homogeneous_sis_infinity_restricted_svp,
    restriction_infinity_norm,
    restriction_infinity_norm_vector,
)
from sisinf.types import Candidate, Instance  # noqa: E402


def _inst() -> Instance:
    return Instance(
        name="small",
        index=0,
        n=2,
        m=3,
        q=7,
        gamma=5,
        A=np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int64),
        t=np.zeros(2, dtype=np.int64),
        require_l2_ge_q=False,
        homogeneous=True,
        source_path=Path("memory"),
    )


def _cand(linf_u: int, linf_v: int) -> Candidate:
    return Candidate(
        u=np.array([linf_u, 0], dtype=np.int64),
        v=np.array([linf_v, 0, 0], dtype=np.int64),
        linf_u=linf_u,
        linf_v=linf_v,
        l2sq=linf_u * linf_u + linf_v * linf_v,
        congruence_ok=True,
        valid_main=True,
        valid_extra=True,
        meta={"max_linf": max(linf_u, linf_v)},
    )


def test_prob_infinity_norm_pass_stays_in_unit_interval() -> None:
    values = [
        prob_infinity_norm_pass(dim=5, bound=3, len_bound=2.0),
        prob_infinity_norm_pass(dim=5, bound=3, len_bound=10.0),
        prob_infinity_norm_pass(dim=100, bound=15, len_bound=80.0),
    ]
    assert all(0.0 <= value <= 1.0 for value in values)


def test_prob_infinity_norm_pass_is_nonincreasing_in_len() -> None:
    dim = 8
    bound = 3
    probs = [prob_infinity_norm_pass(dim=dim, bound=bound, len_bound=len_bound) for len_bound in [2.0, 4.0, 8.0, 16.0]]
    assert probs == sorted(probs, reverse=True)


def test_required_list_size_increases_with_target_success() -> None:
    p_single = 0.2
    sizes = [
        required_list_size(p_success=0.10, p_single=p_single),
        required_list_size(p_success=0.50, p_single=p_single),
        required_list_size(p_success=0.90, p_single=p_single),
    ]
    assert sizes[0] < sizes[1] < sizes[2]


def test_required_list_size_handles_edge_cases() -> None:
    assert required_list_size(p_success=0.0, p_single=0.2) == 0.0
    assert math.isinf(required_list_size(p_success=0.5, p_single=0.0))
    assert required_list_size(p_success=0.5, p_single=1.0) == 1.0


def test_restriction_infinity_norm_matches_bound() -> None:
    assert restriction_infinity_norm(_cand(linf_u=4, linf_v=5), bound=5) is True
    assert restriction_infinity_norm(_cand(linf_u=6, linf_v=5), bound=5) is False
    assert restriction_infinity_norm_vector(np.array([1, -5, 3], dtype=np.int64), bound=5) is True
    assert restriction_infinity_norm_vector(np.array([1, -6, 3], dtype=np.int64), bound=5) is False


def test_make_homogeneous_sis_infinity_restricted_svp_models_instance() -> None:
    inst = _inst()
    model = make_homogeneous_sis_infinity_restricted_svp(inst)

    assert model.dimension == inst.n + inst.m
    assert model.bound == inst.gamma
    assert "heuristic approximation" in model.heuristic_notes[0]
    assert model.restriction_holds(_cand(linf_u=5, linf_v=5)) is True
    assert model.restriction_holds(_cand(linf_u=6, linf_v=5)) is False
    assert 0.0 <= model.related_probability(12.0) <= 1.0
    assert model.required_list_size(p_success=0.8, len_bound=12.0) > 0.0
