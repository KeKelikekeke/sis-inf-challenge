"""Homogeneous SIS infinity-norm LLL/BKZ baseline.

This module intentionally implements only a conservative baseline: construct
the homogeneous SIS row basis, run LLL then BKZ, and validate candidates from
the first few reduced basis rows. It does not implement embedding, random
search, enumeration, or row combinations.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from sisinf.lattice import (
    build_homogeneous_sis_row_basis,
    decode_lattice_vector_to_uv,
    extract_row_vectors,
    integer_matrix_to_numpy,
    to_fpylll_integer_matrix,
)
from sisinf.search import search_homogeneous_candidate_pool, summarize_search_results
from sisinf.types import Candidate, Instance
from sisinf.validate import validate_candidate


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
) -> list[Candidate]:
    """Decode and validate candidates from the first ``top_k`` reduced rows."""

    if not inst.homogeneous:
        raise ValueError(f"{inst.name} is inhomogeneous; homogeneous candidate collection requires homogeneous input")
    if top_k < 0:
        raise ValueError(f"top_k must be non-negative, got {top_k}")

    cands: list[Candidate] = []
    for row in extract_row_vectors(B_row_red, limit=top_k):
        u, v = decode_lattice_vector_to_uv(row, inst)
        cands.append(validate_candidate(inst, u=u, v=v))
    return cands


def collect_reduced_row_vectors(B_row_red: np.ndarray, top_k: int = 20) -> list[np.ndarray]:
    """Collect the first ``top_k`` vectors from a reduced row-basis."""

    if top_k < 0:
        raise ValueError(f"top_k must be non-negative, got {top_k}")
    return extract_row_vectors(B_row_red, limit=top_k)


def solve_homogeneous_bkz_baseline(
    inst: Instance,
    beta: int,
    max_loops: int = 2,
    top_k: int = 20,
) -> list[Candidate]:
    """Run the homogeneous SIS infinity-norm baseline and return validated row candidates."""

    if not inst.homogeneous:
        raise ValueError(f"{inst.name} is inhomogeneous; homogeneous BKZ baseline only accepts homogeneous instances")
    B_row = build_homogeneous_sis_row_basis(inst)
    B_row_lll = run_lll_on_row_basis(B_row)
    B_row_bkz = run_bkz_on_row_basis(B_row_lll, beta=beta, max_loops=max_loops)
    return collect_homogeneous_candidates_from_row_basis(inst, B_row_bkz, top_k=top_k)


def solve_homogeneous_bkz_with_search(
    inst: Instance,
    beta: int,
    max_loops: int = 2,
    top_k: int = 20,
    pair_max_base: int = 20,
    pair_budget: int = 200,
) -> list[Candidate]:
    """Run BKZ baseline and expand reduced rows with lightweight search."""

    if not inst.homogeneous:
        raise ValueError(f"{inst.name} is inhomogeneous; homogeneous BKZ search only accepts homogeneous instances")
    B_row = build_homogeneous_sis_row_basis(inst)
    B_row_lll = run_lll_on_row_basis(B_row)
    B_row_bkz = run_bkz_on_row_basis(B_row_lll, beta=beta, max_loops=max_loops)
    base_vecs = collect_reduced_row_vectors(B_row_bkz, top_k=top_k)
    return search_homogeneous_candidate_pool(
        inst,
        base_vecs,
        base_top_k=top_k,
        pair_max_base=pair_max_base,
        pair_budget=pair_budget,
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
        f"valid_main_count: {valid_main}",
    ]
    for idx, cand in enumerate(cands[: min(total, 10)], start=1):
        lines.append(
            "candidate[{idx}]: linf_u={linf_u}, linf_v={linf_v}, "
            "l2sq={l2sq}, valid_main={valid_main}, valid_extra={valid_extra}".format(
                idx=idx,
                linf_u=cand.linf_u,
                linf_v=cand.linf_v,
                l2sq=cand.l2sq,
                valid_main=cand.valid_main,
                valid_extra=cand.valid_extra,
            )
        )
    return "\n".join(lines)
