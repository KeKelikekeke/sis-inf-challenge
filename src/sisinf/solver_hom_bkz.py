"""Homogeneous SIS infinity-norm LLL/BKZ baseline.

This module intentionally implements only a conservative baseline: construct
the homogeneous SIS row basis, run LLL then BKZ, and validate candidates from
the first few reduced basis rows. It does not implement embedding, random
search, enumeration, or row combinations.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from sisinf.lattice import (
    build_homogeneous_sis_row_basis,
    decode_lattice_vector_to_uv,
    extract_row_vectors,
    integer_matrix_to_numpy,
    to_fpylll_integer_matrix,
)
from sisinf.search import (
    candidate_is_trivial,
    filter_candidates_for_main_pool,
    search_homogeneous_candidate_pool,
    select_search_base_vectors,
    sort_candidates_by_selection_score,
    summarize_candidate_filter_stats,
    summarize_candidate_selection_order,
    summarize_candidate_validation_stats,
    summarize_decoded_vector_stats,
    summarize_search_results,
)
from sisinf.types import Candidate, Instance
from sisinf.validate import validate_candidate

logger = logging.getLogger(__name__)


def _import_fpylll() -> tuple[Any, Any, Any]:
    """Import fpylll lazily and return ``IntegerMatrix``, ``LLL``, and ``BKZ``."""

    try:
        from fpylll import BKZ, LLL, IntegerMatrix
    except ImportError as exc:
        raise ImportError(
            "Homogeneous BKZ baseline requires optional dependency fpylll. "
            "Install it with 'python -m pip install fpylll' or "
            "'conda install -c conda-forge fpylll'."
        ) from exc
    return IntegerMatrix, LLL, BKZ


def run_lll_on_row_basis(B_row: np.ndarray) -> np.ndarray:
    """Run fpylll LLL reduction on a copy of a row-basis matrix."""

    _, LLL, _ = _import_fpylll()
    mat = to_fpylll_integer_matrix(np.asarray(B_row, dtype=np.int64).copy())
    LLL.reduction(mat)
    return integer_matrix_to_numpy(mat)


def run_bkz_on_row_basis(B_row: np.ndarray, beta: int, max_loops: int = 2) -> np.ndarray:
    """Run fpylll BKZ reduction on a copy of a row-basis matrix."""

    if max_loops < 1:
        raise ValueError(f"max_loops must be >= 1, got {max_loops}")
    arr = np.asarray(B_row, dtype=np.int64)
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
        raise ValueError(f"B_row must be a square two-dimensional matrix; got shape {arr.shape}")
    dim = arr.shape[0]
    if beta < 2 or beta > dim:
        raise ValueError(f"beta must satisfy 2 <= beta <= dim ({dim}), got {beta}")

    _, _, BKZ = _import_fpylll()
    mat = to_fpylll_integer_matrix(arr.copy())
    params = BKZ.Param(block_size=int(beta), max_loops=int(max_loops))
    BKZ.reduction(mat, params)
    return integer_matrix_to_numpy(mat)


def collect_homogeneous_candidates_from_row_basis(
    inst: Instance,
    B_row_red: np.ndarray,
    top_k: int = 20,
    filter_trivial_candidates: bool = True,
) -> list[Candidate]:
    """Decode, filter, score, and return the best ``top_k`` reduced-row candidates."""

    if not inst.homogeneous:
        raise ValueError(f"{inst.name} is inhomogeneous; homogeneous candidate collection requires homogeneous input")
    if top_k < 0:
        raise ValueError(f"top_k must be non-negative, got {top_k}")

    rows = extract_row_vectors(B_row_red, limit=None)
    logger.info(summarize_decoded_vector_stats(inst, rows, label="baseline_rows_pre_validation", preview=100))

    cands: list[Candidate] = []
    for row in rows:
        u, v = decode_lattice_vector_to_uv(row, inst)
        cands.append(validate_candidate(inst, u=u, v=v))
    logger.info(
        summarize_candidate_validation_stats(
            cands,
            label="baseline_candidates_post_validation",
            preview=100,
        )
    )
    filtered = filter_candidates_for_main_pool(inst, cands, enabled=filter_trivial_candidates)
    logger.info(
        summarize_candidate_filter_stats(
            inst,
            cands,
            filtered,
            label="baseline_candidate_filter",
            enabled=filter_trivial_candidates,
        )
    )
    logger.info(
        summarize_candidate_validation_stats(
            filtered,
            label="baseline_candidates_post_filter",
            preview=100,
        )
    )
    logger.info(summarize_candidate_selection_order(inst, filtered, label="baseline_selection_pre_sort", preview=20))
    sorted_cands = sort_candidates_by_selection_score(inst, filtered)
    logger.info(summarize_candidate_selection_order(inst, sorted_cands, label="baseline_selection_post_sort", preview=20))
    selected = sorted_cands[:top_k]
    logger.info(
        summarize_candidate_validation_stats(
            selected,
            label="baseline_candidates_selected_top_k",
            preview=100,
        )
    )
    return selected


def collect_reduced_row_vectors(B_row_red: np.ndarray, top_k: int = 20) -> list[np.ndarray]:
    """Collect the first ``top_k`` vectors from a reduced row-basis."""

    if top_k < 0:
        raise ValueError(f"top_k must be non-negative, got {top_k}")
    return extract_row_vectors(B_row_red, limit=top_k)


def collect_scored_reduced_row_vectors(
    inst: Instance,
    B_row_red: np.ndarray,
    top_k: int = 20,
    filter_trivial_candidates: bool = True,
) -> list[np.ndarray]:
    """Select reduced rows with relaxed search-base filtering before expansion."""

    if not inst.homogeneous:
        raise ValueError(f"{inst.name} is inhomogeneous; scored row selection requires homogeneous input")
    if top_k < 0:
        raise ValueError(f"top_k must be non-negative, got {top_k}")

    rows = extract_row_vectors(B_row_red, limit=None)
    return select_search_base_vectors(
        inst,
        rows,
        base_top_k=top_k,
        filter_trivial_candidates=filter_trivial_candidates,
    )


def solve_homogeneous_bkz_baseline(
    inst: Instance,
    beta: int,
    max_loops: int = 2,
    top_k: int = 20,
    filter_trivial_candidates: bool = True,
) -> list[Candidate]:
    """Run the homogeneous SIS infinity-norm baseline and return validated row candidates."""

    if not inst.homogeneous:
        raise ValueError(f"{inst.name} is inhomogeneous; homogeneous BKZ baseline only accepts homogeneous instances")
    B_row = build_homogeneous_sis_row_basis(inst)
    B_row_lll = run_lll_on_row_basis(B_row)
    B_row_bkz = run_bkz_on_row_basis(B_row_lll, beta=beta, max_loops=max_loops)
    bkz_rows_preview = extract_row_vectors(B_row_bkz, limit=min(100, B_row_bkz.shape[0]))
    logger.info(summarize_decoded_vector_stats(inst, bkz_rows_preview, label="bkz_reduced_basis_rows", preview=100))
    return collect_homogeneous_candidates_from_row_basis(
        inst,
        B_row_bkz,
        top_k=top_k,
        filter_trivial_candidates=filter_trivial_candidates,
    )


def solve_homogeneous_bkz_with_search(
    inst: Instance,
    beta: int,
    max_loops: int = 2,
    top_k: int = 20,
    pair_max_base: int = 20,
    pair_budget: int = 200,
    filter_trivial_candidates: bool = True,
    combo_mode: str = "basic",
    combo_max_base: int = 4,
    combo_budget: int = 100,
    include_triples: bool = False,
) -> list[Candidate]:
    """Run BKZ baseline and expand reduced rows with lightweight search."""

    if not inst.homogeneous:
        raise ValueError(f"{inst.name} is inhomogeneous; homogeneous BKZ search only accepts homogeneous instances")
    B_row = build_homogeneous_sis_row_basis(inst)
    B_row_lll = run_lll_on_row_basis(B_row)
    B_row_bkz = run_bkz_on_row_basis(B_row_lll, beta=beta, max_loops=max_loops)
    bkz_rows_preview = extract_row_vectors(B_row_bkz, limit=min(100, B_row_bkz.shape[0]))
    logger.info(summarize_decoded_vector_stats(inst, bkz_rows_preview, label="bkz_reduced_basis_rows", preview=100))
    base_vecs = collect_scored_reduced_row_vectors(
        inst,
        B_row_bkz,
        top_k=top_k,
        filter_trivial_candidates=filter_trivial_candidates,
    )
    return search_homogeneous_candidate_pool(
        inst,
        base_vecs,
        base_top_k=top_k,
        pair_max_base=pair_max_base,
        pair_budget=pair_budget,
        filter_trivial_candidates=filter_trivial_candidates,
        combo_mode=combo_mode,
        combo_max_base=combo_max_base,
        combo_budget=combo_budget,
        include_triples=include_triples,
    )


def summarize_candidate_pool(cands: list[Candidate]) -> str:
    """Summarize a searched candidate pool."""

    return summarize_search_results(cands)


def summarize_candidate_list(cands: list[Candidate]) -> str:
    """Return a compact summary for a list of validated candidates."""

    total = len(cands)
    congruent = sum(1 for cand in cands if cand.congruence_ok)
    valid_main = sum(1 for cand in cands if cand.valid_main)
    lines = [
        f"candidate_count: {total}",
        f"congruence_ok_count: {congruent}",
        f"valid_extra_count: {sum(1 for cand in cands if cand.valid_extra)}",
        f"valid_main_count: {valid_main}",
        f"trivial_candidate_count: {sum(1 for cand in cands if candidate_is_trivial(cand))}",
        f"v_zero_count: {sum(1 for cand in cands if np.all(cand.v == 0))}",
        f"linf_v_zero_count: {sum(1 for cand in cands if cand.linf_v == 0)}",
        f"remaining_nontrivial_count: {sum(1 for cand in cands if not candidate_is_trivial(cand))}",
    ]
    for idx, cand in enumerate(cands[: min(total, 10)], start=1):
        lines.append(
            "candidate[{idx}]: linf_u={linf_u}, linf_v={linf_v}, "
            "l2sq={l2sq}, valid_main={valid_main}, valid_extra={valid_extra}, "
            "v_zero={v_zero}, linf_v_zero={linf_v_zero}".format(
                idx=idx,
                linf_u=cand.linf_u,
                linf_v=cand.linf_v,
                l2sq=cand.l2sq,
                valid_main=cand.valid_main,
                valid_extra=cand.valid_extra,
                v_zero=bool(np.all(cand.v == 0)),
                linf_v_zero=cand.linf_v == 0,
            )
        )
    return "\n".join(lines)
