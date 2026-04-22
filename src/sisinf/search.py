"""Lightweight deterministic candidate-pool search for homogeneous SIS."""

from __future__ import annotations

import itertools
import logging

import numpy as np

from sisinf.metrics import l2sq_int, linf_norm_int
from sisinf.lattice import decode_lattice_vector_to_uv
from sisinf.types import Candidate, Instance
from sisinf.validate import validate_candidate

logger = logging.getLogger(__name__)

SearchComboStats = dict[str, int | bool | str]


def vector_fingerprint(x: np.ndarray) -> tuple[int, ...]:
    """Return a hashable exact-coordinate fingerprint for one integer vector."""

    arr = np.asarray(x, dtype=np.int64).reshape(-1)
    return tuple(int(value) for value in arr)


def dedup_integer_vectors(vecs: list[np.ndarray]) -> list[np.ndarray]:
    """Deduplicate integer vectors by exact coordinates while preserving order."""

    seen: set[tuple[int, ...]] = set()
    out: list[np.ndarray] = []
    for vec in vecs:
        arr = np.asarray(vec, dtype=np.int64).reshape(-1)
        key = vector_fingerprint(arr)
        if key in seen:
            continue
        seen.add(key)
        out.append(arr.copy())
    return out


def dedup_candidates(cands: list[Candidate]) -> list[Candidate]:
    """Deduplicate candidates by the full concatenated ``(u, v)`` coordinates."""

    seen: set[tuple[int, ...]] = set()
    out: list[Candidate] = []
    for cand in cands:
        key = vector_fingerprint(np.concatenate([cand.u, cand.v]))
        if key in seen:
            continue
        seen.add(key)
        out.append(cand)
    return out


def candidate_has_zero_v(cand: Candidate) -> bool:
    """Return whether a candidate has the trivial all-zero ``v`` block."""

    return bool(np.all(np.asarray(cand.v, dtype=np.int64).reshape(-1) == 0))


def candidate_is_trivial(cand: Candidate) -> bool:
    """Return whether a candidate has a trivial ``v`` component."""

    return candidate_has_zero_v(cand) or cand.linf_v == 0


def candidate_exceeds_valid_main_bounds(inst: Instance, cand: Candidate) -> bool:
    """Return whether infinity-norm bounds make ``valid_main`` impossible."""

    return cand.linf_u > inst.gamma or cand.linf_v > inst.gamma


def candidate_filter_reason(inst: Instance, cand: Candidate) -> str | None:
    """Return the main-pool filter reason for a candidate, if any."""

    if candidate_has_zero_v(cand):
        return "v_zero"
    if cand.linf_v == 0:
        return "linf_v_zero"
    if candidate_exceeds_valid_main_bounds(inst, cand):
        return "valid_main_norm_bound_exceeded"
    return None


def summarize_candidate_filter_stats(
    inst: Instance,
    cands_before: list[Candidate],
    cands_after: list[Candidate],
    label: str,
    enabled: bool,
) -> str:
    """Summarize the main-pool candidate filter without mutating candidates."""

    reasons = [candidate_filter_reason(inst, cand) for cand in cands_before]
    trivial_candidate_count = sum(1 for reason in reasons if reason in {"v_zero", "linf_v_zero"})
    boundary_count = sum(1 for reason in reasons if reason == "valid_main_norm_bound_exceeded")
    filtered_out_count = len(cands_before) - len(cands_after)
    remaining_nontrivial_count = sum(1 for cand in cands_after if not candidate_is_trivial(cand))
    return "\n".join(
        [
            f"{label}: filter_enabled={enabled}",
            f"{label}: candidate_count_before={len(cands_before)}",
            f"{label}: trivial_candidate_count={trivial_candidate_count}",
            f"{label}: valid_main_norm_bound_exceeded_count={boundary_count}",
            f"{label}: filtered_out_count={filtered_out_count}",
            f"{label}: remaining_nontrivial_count={remaining_nontrivial_count}",
        ]
    )


def filter_candidates_for_main_pool(
    inst: Instance,
    cands: list[Candidate],
    enabled: bool = True,
) -> list[Candidate]:
    """Filter trivial or obviously impossible candidates before ranking/returning."""

    if not enabled:
        return list(cands)
    return [cand for cand in cands if candidate_filter_reason(inst, cand) is None]


def search_base_filter_reason(cand: Candidate) -> str | None:
    """Return the relaxed search-base filter reason for one candidate."""

    if candidate_has_zero_v(cand):
        return "v_zero"
    if cand.linf_v == 0:
        return "linf_v_zero"
    if cand.l2sq < 0:
        return "abnormal_negative_l2sq"
    return None


def search_base_selection_score(cand: Candidate) -> tuple[int, int]:
    """Score a row by how useful it may be as a combination-search base."""

    return (max(cand.linf_u, cand.linf_v), cand.l2sq)


def summarize_search_base_pool_stats_for_inst(
    inst: Instance,
    cands_before: list[Candidate],
    cands_after: list[Candidate],
    selected: list[Candidate],
    duplicate_row_count: int,
    label: str,
) -> str:
    """Summarize relaxed search-base filtering with instance gamma."""

    total_validated_rows = len(cands_before)
    trivial_row_count = sum(1 for cand in cands_before if candidate_is_trivial(cand))
    nontrivial_row_count = total_validated_rows - trivial_row_count
    norm_exceeded_nontrivial_count = sum(
        1
        for cand in cands_before
        if not candidate_is_trivial(cand) and candidate_exceeds_valid_main_bounds(inst, cand)
    )
    return "\n".join(
        [
            f"{label}: total_validated_rows={total_validated_rows}",
            f"{label}: trivial_row_count={trivial_row_count}",
            f"{label}: nontrivial_row_count={nontrivial_row_count}",
            f"{label}: norm_exceeded_nontrivial_count={norm_exceeded_nontrivial_count}",
            f"{label}: duplicate_row_count={duplicate_row_count}",
            f"{label}: relaxed_remaining_count={len(cands_after)}",
            f"{label}: selected_search_base_count={len(selected)}",
        ]
    )


def summarize_search_base_selection_order(
    cands: list[Candidate],
    label: str,
    preview: int = 20,
) -> str:
    """Summarize selected search-base rows with relaxed score components."""

    if preview < 0:
        raise ValueError(f"preview must be non-negative, got {preview}")
    shown = min(len(cands), preview)
    lines = [f"{label}: selected_search_base_count={len(cands)}, preview_count={shown}"]
    for idx, cand in enumerate(cands[:shown], start=1):
        max_linf, l2sq = search_base_selection_score(cand)
        lines.append(
            "{label}[{idx}]: score_search_base=({max_linf}, {l2sq}), "
            "linf_u={linf_u}, linf_v={linf_v}, l2sq={l2sq}, "
            "v_zero={v_zero}, linf_v_zero={linf_v_zero}, congruence_ok={congruence_ok}, "
            "valid_extra={valid_extra}, valid_main={valid_main}".format(
                label=label,
                idx=idx,
                max_linf=max_linf,
                l2sq=l2sq,
                linf_u=cand.linf_u,
                linf_v=cand.linf_v,
                v_zero=candidate_has_zero_v(cand),
                linf_v_zero=cand.linf_v == 0,
                congruence_ok=cand.congruence_ok,
                valid_extra=cand.valid_extra,
                valid_main=cand.valid_main,
            )
        )
    return "\n".join(lines)


def select_search_base_vector_pairs(
    inst: Instance,
    base_vecs: list[np.ndarray],
    base_top_k: int = 20,
    filter_trivial_candidates: bool = True,
) -> list[tuple[np.ndarray, Candidate]]:
    """Select relaxed nontrivial base vectors for linear-combination search."""

    if base_top_k < 0:
        raise ValueError(f"base_top_k must be non-negative, got {base_top_k}")

    pairs: list[tuple[np.ndarray, Candidate]] = []
    seen_vecs: set[tuple[int, ...]] = set()
    duplicate_row_count = 0
    for vec in base_vecs:
        arr = np.asarray(vec, dtype=np.int64).reshape(-1)
        key = vector_fingerprint(arr)
        if key in seen_vecs:
            duplicate_row_count += 1
            continue
        seen_vecs.add(key)
        u, v = decode_lattice_vector_to_uv(arr, inst)
        pairs.append((arr.copy(), validate_candidate(inst, u=u, v=v)))

    all_cands = [cand for _, cand in pairs]
    logger.info(summarize_decoded_vector_stats(inst, [row for row, _ in pairs], label="search_base_input_vectors", preview=100))
    logger.info(
        summarize_candidate_validation_stats(
            all_cands,
            label="search_base_candidates_pre_relaxed_filter",
            preview=100,
        )
    )

    if filter_trivial_candidates:
        relaxed_pairs = [(row, cand) for row, cand in pairs if search_base_filter_reason(cand) is None]
    else:
        relaxed_pairs = list(pairs)
    relaxed_cands = [cand for _, cand in relaxed_pairs]
    sorted_pairs = sorted(relaxed_pairs, key=lambda item: search_base_selection_score(item[1]))
    selected_pairs = sorted_pairs[:base_top_k]
    selected_cands = [cand for _, cand in selected_pairs]
    logger.info(
        summarize_search_base_pool_stats_for_inst(
            inst,
            all_cands,
            relaxed_cands,
            selected_cands,
            duplicate_row_count,
            label="search_base_pool",
        )
    )
    logger.info(summarize_search_base_selection_order(selected_cands, label="selected_search_base", preview=20))
    return selected_pairs


def candidate_valid_main_gap(inst: Instance, cand: Candidate) -> int:
    """Return a small penalty for missing pieces of the ``valid_main`` condition."""

    congruence_gap = 0 if cand.congruence_ok else 1
    linf_u_excess = max(0, cand.linf_u - inst.gamma)
    linf_v_excess = max(0, cand.linf_v - inst.gamma)
    return congruence_gap + linf_u_excess + linf_v_excess


def candidate_selection_score(inst: Instance, cand: Candidate) -> tuple[int, int, int]:
    """Return the simple constraint-oriented candidate selection score."""

    return (
        max(cand.linf_u, cand.linf_v),
        cand.l2sq,
        candidate_valid_main_gap(inst, cand),
    )


def sort_candidates_by_selection_score(inst: Instance, cands: list[Candidate]) -> list[Candidate]:
    """Stably sort candidates by the main-pool selection score."""

    return sorted(cands, key=lambda cand: candidate_selection_score(inst, cand))


def summarize_candidate_selection_order(
    inst: Instance,
    cands: list[Candidate],
    label: str,
    preview: int = 20,
) -> str:
    """Summarize candidate order with score components for selection debugging."""

    if preview < 0:
        raise ValueError(f"preview must be non-negative, got {preview}")
    shown = min(len(cands), preview)
    lines = [f"{label}: candidate_count={len(cands)}, preview_count={shown}"]
    for idx, cand in enumerate(cands[:shown], start=1):
        max_linf, l2sq, valid_main_gap = candidate_selection_score(inst, cand)
        lines.append(
            "{label}[{idx}]: score=({max_linf}, {l2sq}, {valid_main_gap}), "
            "max_linf={max_linf}, l2sq={l2sq}, valid_main_gap={valid_main_gap}, "
            "linf_u={linf_u}, linf_v={linf_v}, v_zero={v_zero}, linf_v_zero={linf_v_zero}, "
            "congruence_ok={congruence_ok}, valid_extra={valid_extra}, valid_main={valid_main}".format(
                label=label,
                idx=idx,
                max_linf=max_linf,
                l2sq=l2sq,
                valid_main_gap=valid_main_gap,
                linf_u=cand.linf_u,
                linf_v=cand.linf_v,
                v_zero=candidate_has_zero_v(cand),
                linf_v_zero=cand.linf_v == 0,
                congruence_ok=cand.congruence_ok,
                valid_extra=cand.valid_extra,
                valid_main=cand.valid_main,
            )
        )
    return "\n".join(lines)


def summarize_search_base_vector_stats(
    inst: Instance,
    candidate_base_count: int,
    selected_base_vecs: list[np.ndarray],
    base_top_k: int,
    pair_max_base: int,
    pair_budget: int,
    generated_pair_count: int,
    combo_mode: str,
    combo_max_base: int,
    combo_budget: int,
    generated_combo_count: int,
    label: str,
) -> str:
    """Summarize which base vectors enter pairwise search and budget use."""

    selected_count = len(selected_base_vecs)
    v_nonzero_count = 0
    for vec in selected_base_vecs:
        _, v = decode_lattice_vector_to_uv(vec, inst)
        if not bool(np.all(v == 0)):
            v_nonzero_count += 1
    v_nonzero_ratio = 0.0 if selected_count == 0 else v_nonzero_count / selected_count
    effective_pair_base_count = min(pair_max_base, selected_count)
    return "\n".join(
        [
            f"{label}: candidate_base_vector_count={candidate_base_count}",
            f"{label}: selected_base_vector_count={selected_count}",
            f"{label}: base_top_k={base_top_k}",
            f"{label}: selected_base_v_nonzero_count={v_nonzero_count}",
            f"{label}: selected_base_v_nonzero_ratio={v_nonzero_ratio:.6f}",
            f"{label}: pair_max_base_requested={pair_max_base}",
            f"{label}: pair_base_count_used={effective_pair_base_count}",
            f"{label}: pair_budget_requested={pair_budget}",
            f"{label}: generated_pair_count={generated_pair_count}",
            f"{label}: pair_budget_exhausted={generated_pair_count >= pair_budget if pair_budget > 0 else False}",
            f"{label}: combo_mode={combo_mode}",
            f"{label}: combo_max_base_requested={combo_max_base}",
            f"{label}: combo_base_count_used={min(combo_max_base, selected_count)}",
            f"{label}: combo_budget_requested={combo_budget}",
            f"{label}: generated_combo_count={generated_combo_count}",
            f"{label}: combo_budget_exhausted={generated_combo_count >= combo_budget if combo_budget > 0 else False}",
        ]
    )


def select_search_base_vectors(
    inst: Instance,
    base_vecs: list[np.ndarray],
    base_top_k: int = 20,
    filter_trivial_candidates: bool = True,
) -> list[np.ndarray]:
    """Select nontrivial, scored base vectors for pairwise search expansion."""

    return [
        row
        for row, _ in select_search_base_vector_pairs(
            inst,
            base_vecs,
            base_top_k=base_top_k,
            filter_trivial_candidates=filter_trivial_candidates,
        )
    ]


def summarize_decoded_vector_stats(
    inst: Instance,
    vecs: list[np.ndarray],
    label: str,
    preview: int = 100,
) -> str:
    """Summarize decoded ``[u; v]`` vectors before full candidate validation."""

    if preview < 0:
        raise ValueError(f"preview must be non-negative, got {preview}")
    shown = min(len(vecs), preview)
    decoded: list[tuple[int, int, int, bool]] = []
    for vec in vecs[:shown]:
        u, v = decode_lattice_vector_to_uv(vec, inst)
        linf_u = linf_norm_int(u)
        linf_v = linf_norm_int(v)
        l2sq = l2sq_int(u) + l2sq_int(v)
        v_zero = bool(np.all(v == 0))
        decoded.append((linf_u, linf_v, l2sq, v_zero))

    lines = [
        f"{label}: vector_count={len(vecs)}, preview_count={shown}",
        f"{label}: preview_v_zero_count={sum(1 for _, _, _, v_zero in decoded if v_zero)}",
        f"{label}: preview_linf_v_zero_count={sum(1 for _, linf_v, _, _ in decoded if linf_v == 0)}",
    ]
    for idx, (linf_u, linf_v, l2sq, v_zero) in enumerate(decoded, start=1):
        lines.append(
            "{label}[{idx}]: linf_u={linf_u}, linf_v={linf_v}, l2sq={l2sq}, "
            "v_zero={v_zero}, linf_v_zero={linf_v_zero}".format(
                label=label,
                idx=idx,
                linf_u=linf_u,
                linf_v=linf_v,
                l2sq=l2sq,
                v_zero=v_zero,
                linf_v_zero=linf_v == 0,
            )
        )
    return "\n".join(lines)


def summarize_candidate_validation_stats(
    cands: list[Candidate],
    label: str,
    preview: int = 100,
) -> str:
    """Summarize validated candidate counts and leading candidate metrics."""

    if preview < 0:
        raise ValueError(f"preview must be non-negative, got {preview}")
    shown = min(len(cands), preview)
    lines = [
        f"{label}: candidate_count={len(cands)}, preview_count={shown}",
        f"{label}: congruence_ok_count={sum(1 for cand in cands if cand.congruence_ok)}",
        f"{label}: valid_extra_count={sum(1 for cand in cands if cand.valid_extra)}",
        f"{label}: valid_main_count={sum(1 for cand in cands if cand.valid_main)}",
        f"{label}: trivial_candidate_count={sum(1 for cand in cands if candidate_is_trivial(cand))}",
        f"{label}: v_zero_count={sum(1 for cand in cands if candidate_has_zero_v(cand))}",
        f"{label}: linf_v_zero_count={sum(1 for cand in cands if cand.linf_v == 0)}",
        f"{label}: remaining_nontrivial_count={sum(1 for cand in cands if not candidate_is_trivial(cand))}",
    ]
    for idx, cand in enumerate(cands[:shown], start=1):
        lines.append(
            "{label}[{idx}]: linf_u={linf_u}, linf_v={linf_v}, l2sq={l2sq}, "
            "v_zero={v_zero}, linf_v_zero={linf_v_zero}, congruence_ok={congruence_ok}, "
            "valid_extra={valid_extra}, valid_main={valid_main}".format(
                label=label,
                idx=idx,
                linf_u=cand.linf_u,
                linf_v=cand.linf_v,
                l2sq=cand.l2sq,
                v_zero=candidate_has_zero_v(cand),
                linf_v_zero=cand.linf_v == 0,
                congruence_ok=cand.congruence_ok,
                valid_extra=cand.valid_extra,
                valid_main=cand.valid_main,
            )
        )
    return "\n".join(lines)


def rank_candidates(cands: list[Candidate]) -> list[Candidate]:
    """Return candidates sorted by validity and simple norm-based priorities."""

    return sorted(
        cands,
        key=lambda cand: (
            not cand.valid_main,
            not cand.valid_extra,
            max(cand.linf_u, cand.linf_v),
            cand.linf_u + cand.linf_v,
            cand.l2sq,
        ),
    )


def generate_pairwise_combinations(
    base_vecs: list[np.ndarray],
    max_base: int = 20,
    pair_budget: int = 200,
    include_negations: bool = True,
) -> list[np.ndarray]:
    """Generate bounded two-vector combinations with coefficients in ``{-1, 0, 1}``.

    Only the first ``max_base`` vectors are considered, only pairs ``i < j`` are
    combined, and ``pair_budget`` caps the number of unique generated vectors.
    Degenerate zero vectors and outputs identical to an input single vector are
    discarded.
    """

    if max_base < 0:
        raise ValueError(f"max_base must be non-negative, got {max_base}")
    if pair_budget < 0:
        raise ValueError(f"pair_budget must be non-negative, got {pair_budget}")
    if pair_budget == 0 or max_base == 0:
        return []

    base = [np.asarray(vec, dtype=np.int64).reshape(-1) for vec in base_vecs[:max_base]]
    if not base:
        return []
    dim = base[0].shape
    for idx, vec in enumerate(base):
        if vec.shape != dim:
            raise ValueError(f"base_vecs[{idx}] has shape {vec.shape}; expected {dim}")

    single_keys = {vector_fingerprint(vec) for vec in base}
    out: list[np.ndarray] = []
    seen: set[tuple[int, ...]] = set()
    coeff_pairs = [(1, 1), (1, -1)]
    if include_negations:
        coeff_pairs.extend([(-1, 1), (-1, -1)])

    for i in range(len(base)):
        for j in range(i + 1, len(base)):
            for ci, cj in coeff_pairs:
                combo = ci * base[i] + cj * base[j]
                key = vector_fingerprint(combo)
                if not key or all(value == 0 for value in key):
                    continue
                if key in single_keys or key in seen:
                    continue
                seen.add(key)
                out.append(combo.astype(np.int64, copy=True))
                if len(out) >= pair_budget:
                    return out
    return out


def generate_small_coefficient_combinations(
    base_vecs: list[np.ndarray],
    max_base: int = 4,
    combo_budget: int = 100,
    include_triples: bool = False,
) -> tuple[list[np.ndarray], SearchComboStats]:
    """Generate bounded combinations over coefficients ``{-2, -1, 0, 1, 2}``."""

    if max_base < 0:
        raise ValueError(f"max_base must be non-negative, got {max_base}")
    if combo_budget < 0:
        raise ValueError(f"combo_budget must be non-negative, got {combo_budget}")
    if combo_budget == 0 or max_base == 0:
        return [], {
            "mode": "small_coeff",
            "base_count_used": 0,
            "attempted_combination_count": 0,
            "generated_combo_count": 0,
            "duplicate_or_single_count": 0,
            "zero_combo_count": 0,
            "triple_enabled": include_triples,
            "budget_exhausted": False,
        }

    base = [np.asarray(vec, dtype=np.int64).reshape(-1) for vec in base_vecs[:max_base]]
    if not base:
        return [], {
            "mode": "small_coeff",
            "base_count_used": 0,
            "attempted_combination_count": 0,
            "generated_combo_count": 0,
            "duplicate_or_single_count": 0,
            "zero_combo_count": 0,
            "triple_enabled": include_triples,
            "budget_exhausted": False,
        }

    dim = base[0].shape
    for idx, vec in enumerate(base):
        if vec.shape != dim:
            raise ValueError(f"base_vecs[{idx}] has shape {vec.shape}; expected {dim}")

    single_keys = {vector_fingerprint(vec) for vec in base}
    coeffs = [-2, -1, 1, 2]
    out: list[np.ndarray] = []
    seen: set[tuple[int, ...]] = set()
    attempted = 0
    duplicate_or_single = 0
    zero_combo = 0

    def maybe_add(combo: np.ndarray) -> bool:
        nonlocal duplicate_or_single, zero_combo
        key = vector_fingerprint(combo)
        if not key or all(value == 0 for value in key):
            zero_combo += 1
            return False
        if key in single_keys or key in seen:
            duplicate_or_single += 1
            return False
        seen.add(key)
        out.append(combo.astype(np.int64, copy=True))
        return len(out) >= combo_budget

    for i, j in itertools.combinations(range(len(base)), 2):
        for ci, cj in itertools.product(coeffs, repeat=2):
            attempted += 1
            if maybe_add(ci * base[i] + cj * base[j]):
                return out, {
                    "mode": "small_coeff",
                    "base_count_used": len(base),
                    "attempted_combination_count": attempted,
                    "generated_combo_count": len(out),
                    "duplicate_or_single_count": duplicate_or_single,
                    "zero_combo_count": zero_combo,
                    "triple_enabled": include_triples,
                    "budget_exhausted": True,
                }

    if include_triples:
        for i, j, k in itertools.combinations(range(len(base)), 3):
            for ci, cj, ck in itertools.product(coeffs, repeat=3):
                attempted += 1
                if maybe_add(ci * base[i] + cj * base[j] + ck * base[k]):
                    return out, {
                        "mode": "small_coeff",
                        "base_count_used": len(base),
                        "attempted_combination_count": attempted,
                        "generated_combo_count": len(out),
                        "duplicate_or_single_count": duplicate_or_single,
                        "zero_combo_count": zero_combo,
                        "triple_enabled": include_triples,
                        "budget_exhausted": True,
                    }

    return out, {
        "mode": "small_coeff",
        "base_count_used": len(base),
        "attempted_combination_count": attempted,
        "generated_combo_count": len(out),
        "duplicate_or_single_count": duplicate_or_single,
        "zero_combo_count": zero_combo,
        "triple_enabled": include_triples,
        "budget_exhausted": False,
    }


def summarize_combo_generation_stats(
    basic_vecs: list[np.ndarray],
    combo_vecs: list[np.ndarray],
    combo_stats: SearchComboStats,
    label: str,
) -> str:
    """Summarize enhanced combination enumeration and vector diversity."""

    basic_keys = {vector_fingerprint(vec) for vec in basic_vecs}
    combo_keys = {vector_fingerprint(vec) for vec in combo_vecs}
    combined_keys = basic_keys | combo_keys
    new_keys = combo_keys - basic_keys
    return "\n".join(
        [
            f"{label}: mode={combo_stats.get('mode')}",
            f"{label}: triple_enabled={combo_stats.get('triple_enabled')}",
            f"{label}: base_count_used={combo_stats.get('base_count_used')}",
            f"{label}: attempted_combination_count={combo_stats.get('attempted_combination_count')}",
            f"{label}: generated_combo_count={combo_stats.get('generated_combo_count')}",
            f"{label}: duplicate_or_single_count={combo_stats.get('duplicate_or_single_count')}",
            f"{label}: zero_combo_count={combo_stats.get('zero_combo_count')}",
            f"{label}: budget_exhausted={combo_stats.get('budget_exhausted')}",
            f"{label}: basic_unique_vector_count={len(basic_keys)}",
            f"{label}: combo_unique_vector_count={len(combo_keys)}",
            f"{label}: combined_unique_vector_count={len(combined_keys)}",
            f"{label}: new_unique_vs_basic_count={len(new_keys)}",
            f"{label}: diversity_improved={len(new_keys) > 0}",
        ]
    )


def decode_and_validate_vectors(inst: Instance, vecs: list[np.ndarray]) -> list[Candidate]:
    """Decode lattice vectors as ``[u; v]`` and validate each candidate."""

    cands: list[Candidate] = []
    for vec in vecs:
        u, v = decode_lattice_vector_to_uv(vec, inst)
        cands.append(validate_candidate(inst, u=u, v=v))
    return cands


def search_homogeneous_candidate_pool(
    inst: Instance,
    base_vecs: list[np.ndarray],
    base_top_k: int = 20,
    pair_max_base: int = 20,
    pair_budget: int = 200,
    filter_trivial_candidates: bool = True,
    combo_mode: str = "basic",
    combo_max_base: int = 4,
    combo_budget: int = 100,
    include_triples: bool = False,
) -> list[Candidate]:
    """Expand reduced-row candidates with bounded pairwise combinations."""

    if not inst.homogeneous:
        raise ValueError(f"{inst.name} is inhomogeneous; homogeneous candidate-pool search requires homogeneous input")
    if base_top_k < 0:
        raise ValueError(f"base_top_k must be non-negative, got {base_top_k}")
    if combo_mode not in {"basic", "small-coeff"}:
        raise ValueError(f"combo_mode must be 'basic' or 'small-coeff', got {combo_mode!r}")
    if combo_max_base < 0:
        raise ValueError(f"combo_max_base must be non-negative, got {combo_max_base}")
    if combo_budget < 0:
        raise ValueError(f"combo_budget must be non-negative, got {combo_budget}")

    singles = select_search_base_vectors(
        inst,
        base_vecs,
        base_top_k=base_top_k,
        filter_trivial_candidates=filter_trivial_candidates,
    )
    logger.info(summarize_decoded_vector_stats(inst, singles, label="search_singles_pre_validation", preview=100))
    pairwise = generate_pairwise_combinations(
        singles,
        max_base=pair_max_base,
        pair_budget=pair_budget,
        include_negations=True,
    )
    combo_vecs: list[np.ndarray] = []
    combo_stats: SearchComboStats = {
        "mode": combo_mode,
        "base_count_used": 0,
        "attempted_combination_count": 0,
        "generated_combo_count": 0,
        "duplicate_or_single_count": 0,
        "zero_combo_count": 0,
        "triple_enabled": include_triples,
        "budget_exhausted": False,
    }
    if combo_mode == "small-coeff":
        combo_vecs, combo_stats = generate_small_coefficient_combinations(
            singles,
            max_base=combo_max_base,
            combo_budget=combo_budget,
            include_triples=include_triples,
        )
    logger.info(summarize_combo_generation_stats(pairwise, combo_vecs, combo_stats, label="search_combo_generation"))
    logger.info(
        summarize_search_base_vector_stats(
            inst,
            candidate_base_count=len(base_vecs),
            selected_base_vecs=singles,
            base_top_k=base_top_k,
            pair_max_base=pair_max_base,
            pair_budget=pair_budget,
            generated_pair_count=len(pairwise),
            combo_mode=combo_mode,
            combo_max_base=combo_max_base,
            combo_budget=combo_budget,
            generated_combo_count=len(combo_vecs),
            label="search_pair_budget",
        )
    )
    pairwise_cands = decode_and_validate_vectors(inst, pairwise)
    logger.info(
        summarize_candidate_validation_stats(
            pairwise_cands,
            label="search_pairwise_candidates_post_validation",
            preview=100,
        )
    )
    combo_cands = decode_and_validate_vectors(inst, combo_vecs)
    logger.info(
        summarize_candidate_validation_stats(
            combo_cands,
            label="search_combo_candidates_post_validation",
            preview=100,
        )
    )
    logger.info(
        "search_combo_nontrivial: candidate_count=%s, nontrivial_candidate_count=%s",
        len(combo_cands),
        sum(1 for cand in combo_cands if not candidate_is_trivial(cand)),
    )
    raw_vecs = [*singles, *pairwise, *combo_vecs]
    logger.info(
        "search_vector_dedup: before=%s, singles=%s, pairwise=%s, combo=%s",
        len(raw_vecs),
        len(singles),
        len(pairwise),
        len(combo_vecs),
    )
    vecs = dedup_integer_vectors(raw_vecs)
    logger.info("search_vector_dedup: after=%s, removed=%s", len(vecs), len(raw_vecs) - len(vecs))
    logger.info(summarize_decoded_vector_stats(inst, vecs, label="search_vectors_post_dedup_pre_validation", preview=100))
    cands = decode_and_validate_vectors(inst, vecs)
    logger.info(
        summarize_candidate_validation_stats(
            cands,
            label="search_candidates_post_validation_pre_candidate_dedup",
            preview=100,
        )
    )
    filtered = filter_candidates_for_main_pool(inst, cands, enabled=filter_trivial_candidates)
    logger.info(
        summarize_candidate_filter_stats(
            inst,
            cands,
            filtered,
            label="search_candidate_filter",
            enabled=filter_trivial_candidates,
        )
    )
    logger.info(
        summarize_candidate_validation_stats(
            filtered,
            label="search_candidates_post_filter_pre_candidate_dedup",
            preview=100,
        )
    )
    deduped = dedup_candidates(filtered)
    logger.info(
        "search_candidate_dedup: before=%s, after=%s, removed=%s",
        len(filtered),
        len(deduped),
        len(filtered) - len(deduped),
    )
    logger.info(
        summarize_candidate_validation_stats(
            deduped,
            label="search_candidates_post_candidate_dedup_pre_rank",
            preview=100,
        )
    )
    logger.info(summarize_candidate_selection_order(inst, deduped, label="search_selection_pre_sort", preview=20))
    ranked = sort_candidates_by_selection_score(inst, deduped)
    logger.info(summarize_candidate_selection_order(inst, ranked, label="search_selection_post_sort", preview=20))
    logger.info(summarize_candidate_validation_stats(ranked, label="search_candidates_post_rank", preview=100))
    return ranked


def summarize_search_results(cands: list[Candidate], preview: int = 10) -> str:
    """Format a compact summary of ranked search candidates."""

    if preview < 0:
        raise ValueError(f"preview must be non-negative, got {preview}")
    total = len(cands)
    lines = [
        f"candidate_count: {total}",
        f"congruence_ok_count: {sum(1 for cand in cands if cand.congruence_ok)}",
        f"valid_main_count: {sum(1 for cand in cands if cand.valid_main)}",
        f"valid_extra_count: {sum(1 for cand in cands if cand.valid_extra)}",
        f"trivial_candidate_count: {sum(1 for cand in cands if candidate_is_trivial(cand))}",
        f"v_zero_count: {sum(1 for cand in cands if candidate_has_zero_v(cand))}",
        f"linf_v_zero_count: {sum(1 for cand in cands if cand.linf_v == 0)}",
        f"remaining_nontrivial_count: {sum(1 for cand in cands if not candidate_is_trivial(cand))}",
    ]
    for idx, cand in enumerate(cands[: min(total, preview)], start=1):
        max_linf = max(cand.linf_u, cand.linf_v)
        lines.append(
            "candidate[{idx}]: linf_u={linf_u}, linf_v={linf_v}, "
            "l2sq={l2sq}, max_linf={max_linf}, valid_main={valid_main}, "
            "valid_extra={valid_extra}, v_zero={v_zero}, linf_v_zero={linf_v_zero}".format(
                idx=idx,
                linf_u=cand.linf_u,
                linf_v=cand.linf_v,
                l2sq=cand.l2sq,
                max_linf=max_linf,
                valid_main=cand.valid_main,
                valid_extra=cand.valid_extra,
                v_zero=candidate_has_zero_v(cand),
                linf_v_zero=cand.linf_v == 0,
            )
        )
    return "\n".join(lines)
