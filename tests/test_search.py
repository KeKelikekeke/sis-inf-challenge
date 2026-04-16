from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sisinf.search import (  # noqa: E402
    dedup_candidates,
    dedup_integer_vectors,
    generate_pairwise_combinations,
    rank_candidates,
    search_homogeneous_candidate_pool,
    vector_fingerprint,
)
from sisinf.types import Candidate, Instance  # noqa: E402


def _candidate(
    u: list[int],
    v: list[int],
    linf_u: int,
    linf_v: int,
    l2sq: int,
    valid_main: bool,
    valid_extra: bool,
) -> Candidate:
    """Build a compact artificial candidate for search tests."""

    return Candidate(
        u=np.array(u, dtype=np.int64),
        v=np.array(v, dtype=np.int64),
        linf_u=linf_u,
        linf_v=linf_v,
        l2sq=l2sq,
        congruence_ok=valid_main,
        valid_main=valid_main,
        valid_extra=valid_extra,
        meta={"max_linf": max(linf_u, linf_v)},
    )


def _small_inst() -> Instance:
    """Build a tiny homogeneous instance."""

    return Instance(
        name="small",
        index=0,
        n=2,
        m=2,
        q=7,
        gamma=10,
        A=np.array([[1, 2], [3, 4]], dtype=np.int64),
        t=np.zeros(2, dtype=np.int64),
        require_l2_ge_q=False,
        homogeneous=True,
        source_path=Path("memory"),
    )


def _lattice_vec(inst: Instance, w: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Create an exact lattice vector [q w - A v; v]."""

    u = inst.q * w - inst.A @ v
    return np.concatenate([u, v]).astype(np.int64)


def test_dedup_integer_vectors_preserves_first_order() -> None:
    """Identical vectors are kept once in first-seen order."""

    vecs = [np.array([1, 2]), np.array([3, 4]), np.array([1, 2]), np.array([5, 6])]
    out = dedup_integer_vectors(vecs)
    assert [vector_fingerprint(vec) for vec in out] == [(1, 2), (3, 4), (5, 6)]


def test_dedup_candidates_by_u_v() -> None:
    """Candidates with identical (u, v) coordinates are deduplicated."""

    first = _candidate([1, 0], [0, 1], 1, 1, 2, True, True)
    duplicate = _candidate([1, 0], [0, 1], 9, 9, 99, False, False)
    other = _candidate([2, 0], [0, 1], 2, 1, 5, True, True)
    out = dedup_candidates([first, duplicate, other])
    assert out[0] is first
    assert out[1] is other


def test_rank_candidates_uses_fixed_priority_order() -> None:
    """Candidate ranking prioritizes validity and then simple norm metrics."""

    bad = _candidate([0], [0], 0, 0, 0, False, True)
    valid_large = _candidate([4], [1], 4, 1, 17, True, True)
    valid_small_l2 = _candidate([2], [2], 2, 2, 8, True, True)
    valid_same_linf_worse_sum = _candidate([2], [3], 2, 3, 13, True, True)
    no_extra = _candidate([1], [1], 1, 1, 2, True, False)
    ranked = rank_candidates([bad, valid_large, valid_same_linf_worse_sum, no_extra, valid_small_l2])
    assert [id(cand) for cand in ranked] == [
        id(valid_small_l2),
        id(valid_same_linf_worse_sum),
        id(valid_large),
        id(no_extra),
        id(bad),
    ]


def test_generate_pairwise_combinations_is_bounded_and_deduped() -> None:
    """Pairwise generation includes expected sign combinations and obeys budget."""

    x1 = np.array([1, 0], dtype=np.int64)
    x2 = np.array([0, 1], dtype=np.int64)
    x3 = np.array([1, 1], dtype=np.int64)
    combos = generate_pairwise_combinations([x1, x2, x3], max_base=3, pair_budget=10)
    keys = {vector_fingerprint(vec) for vec in combos}
    assert (1, 1) not in keys
    assert (1, -1) in keys
    assert (-1, 1) in keys
    assert (-1, -1) in keys
    assert len(keys) == len(combos)

    limited = generate_pairwise_combinations([x1, x2, x3], max_base=3, pair_budget=2)
    assert len(limited) == 2


def test_search_homogeneous_candidate_pool_returns_sorted_unique_candidates() -> None:
    """Candidate-pool search decodes, validates, deduplicates, and ranks outputs."""

    inst = _small_inst()
    base_vecs = [
        _lattice_vec(inst, np.array([0, 0]), np.array([1, 0])),
        _lattice_vec(inst, np.array([0, 0]), np.array([0, 1])),
        _lattice_vec(inst, np.array([0, 0]), np.array([1, 0])),
    ]
    cands = search_homogeneous_candidate_pool(
        inst,
        base_vecs,
        base_top_k=3,
        pair_max_base=2,
        pair_budget=4,
    )
    assert cands
    assert all(isinstance(cand, Candidate) for cand in cands)
    assert all(cand.u.shape == (inst.n,) and cand.v.shape == (inst.m,) for cand in cands)
    keys = [vector_fingerprint(np.concatenate([cand.u, cand.v])) for cand in cands]
    assert len(keys) == len(set(keys))
    reranked = rank_candidates(cands)
    assert [id(cand) for cand in cands] == [id(cand) for cand in reranked]
