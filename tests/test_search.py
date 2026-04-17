from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sisinf.search import (  # noqa: E402
    dedup_candidates,
    dedup_integer_vectors,
    filter_candidates_for_main_pool,
    generate_pairwise_combinations,
    generate_small_coefficient_combinations,
    rank_candidates,
    search_homogeneous_candidate_pool,
    sort_candidates_by_selection_score,
    vector_fingerprint,
)
from sisinf.solver_hom_bkz import collect_homogeneous_candidates_from_row_basis  # noqa: E402
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


def test_filter_candidates_for_main_pool_removes_trivial_and_impossible() -> None:
    """The main-pool filter removes zero-v and norm-impossible candidates only."""

    inst = _small_inst()
    trivial = _candidate([1, 0], [0, 0], 1, 0, 1, True, True)
    impossible = _candidate([11, 0], [1, 0], 11, 1, 122, False, True)
    promising = _candidate([2, 0], [1, 0], 2, 1, 5, True, True)

    filtered = filter_candidates_for_main_pool(inst, [trivial, impossible, promising])
    assert filtered == [promising]

    disabled = filter_candidates_for_main_pool(inst, [trivial, impossible, promising], enabled=False)
    assert disabled == [trivial, impossible, promising]


def test_sort_candidates_by_selection_score_is_stable_and_repeatable() -> None:
    """Selection scoring is deterministic and preserves ties in input order."""

    inst = _small_inst()
    first_tie = _candidate([2, 0], [1, 0], 2, 1, 5, True, True)
    second_tie = _candidate([0, 2], [0, 1], 2, 1, 5, True, True)
    smaller_max_linf = _candidate([1, 0], [1, 0], 1, 1, 2, True, True)
    larger_l2 = _candidate([2, 0], [2, 0], 2, 2, 8, True, True)

    cands = [first_tie, larger_l2, second_tie, smaller_max_linf]
    ranked_once = sort_candidates_by_selection_score(inst, cands)
    ranked_twice = sort_candidates_by_selection_score(inst, cands)

    assert ranked_once == ranked_twice
    assert ranked_once == [smaller_max_linf, first_tie, second_tie, larger_l2]


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


def test_generate_small_coefficient_combinations_obeys_budget() -> None:
    """Enhanced small-coefficient generation is bounded and deterministic."""

    x1 = np.array([1, 0], dtype=np.int64)
    x2 = np.array([0, 1], dtype=np.int64)
    combos, stats = generate_small_coefficient_combinations([x1, x2], max_base=2, combo_budget=3)

    assert len(combos) == 3
    assert stats["generated_combo_count"] == 3
    assert stats["attempted_combination_count"] == 3
    assert stats["budget_exhausted"] is True


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
    reranked = sort_candidates_by_selection_score(inst, cands)
    assert [id(cand) for cand in cands] == [id(cand) for cand in reranked]


def test_search_homogeneous_candidate_pool_filters_trivial_by_default() -> None:
    """Default search filtering keeps nontrivial rows out of a zero-v prefix."""

    inst = _small_inst()
    base_vecs = [
        _lattice_vec(inst, np.array([1, 0]), np.array([0, 0])),
        _lattice_vec(inst, np.array([0, 0]), np.array([1, 0])),
    ]

    cands = search_homogeneous_candidate_pool(
        inst,
        base_vecs,
        base_top_k=2,
        pair_max_base=2,
        pair_budget=0,
    )
    assert cands
    assert all(cand.linf_v > 0 for cand in cands)

    unfiltered = search_homogeneous_candidate_pool(
        inst,
        base_vecs,
        base_top_k=2,
        pair_max_base=2,
        pair_budget=0,
        filter_trivial_candidates=False,
    )
    assert any(cand.linf_v == 0 for cand in unfiltered)


def test_search_pairwise_uses_scored_nontrivial_base_vectors() -> None:
    """Pairwise expansion uses filtered/scored bases instead of the natural prefix."""

    inst = _small_inst()
    base_vecs = [
        _lattice_vec(inst, np.array([1, 0]), np.array([0, 0])),
        _lattice_vec(inst, np.array([0, 0]), np.array([3, 0])),
        _lattice_vec(inst, np.array([0, 0]), np.array([1, 0])),
        _lattice_vec(inst, np.array([0, 0]), np.array([0, 1])),
    ]

    cands = search_homogeneous_candidate_pool(
        inst,
        base_vecs,
        base_top_k=2,
        pair_max_base=2,
        pair_budget=4,
    )
    candidate_vs = {vector_fingerprint(cand.v) for cand in cands}

    assert (1, 1) in candidate_vs
    assert all(cand.linf_v > 0 for cand in cands)


def test_search_small_coeff_mode_adds_bounded_candidate_diversity() -> None:
    """Small-coeff mode can add candidates outside the basic {-1,0,1} pair set."""

    inst = _small_inst()
    base_vecs = [
        _lattice_vec(inst, np.array([0, 0]), np.array([1, 0])),
        _lattice_vec(inst, np.array([0, 0]), np.array([0, 1])),
    ]

    basic = search_homogeneous_candidate_pool(
        inst,
        base_vecs,
        base_top_k=2,
        pair_max_base=2,
        pair_budget=4,
    )
    enhanced = search_homogeneous_candidate_pool(
        inst,
        base_vecs,
        base_top_k=2,
        pair_max_base=2,
        pair_budget=4,
        combo_mode="small-coeff",
        combo_max_base=2,
        combo_budget=20,
    )

    basic_vs = {vector_fingerprint(cand.v) for cand in basic}
    enhanced_vs = {vector_fingerprint(cand.v) for cand in enhanced}

    assert (2, 1) in enhanced_vs
    assert (2, 1) not in basic_vs
    assert len(enhanced_vs) > len(basic_vs)


def test_reduced_row_selection_scores_before_top_k() -> None:
    """A later nontrivial low-score row can displace earlier trivial/worse rows."""

    inst = _small_inst()
    rows = np.array(
        [
            _lattice_vec(inst, np.array([1, 0]), np.array([0, 0])),
            _lattice_vec(inst, np.array([0, 0]), np.array([3, 0])),
            _lattice_vec(inst, np.array([0, 0]), np.array([1, 0])),
        ],
        dtype=np.int64,
    )

    cands = collect_homogeneous_candidates_from_row_basis(inst, rows, top_k=1)

    assert len(cands) == 1
    assert np.array_equal(cands[0].v, np.array([1, 0], dtype=np.int64))
    assert cands[0].linf_v > 0
