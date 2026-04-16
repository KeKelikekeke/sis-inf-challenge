"""Lightweight deterministic candidate-pool search for homogeneous SIS."""

from __future__ import annotations

import numpy as np

from sisinf.lattice import decode_lattice_vector_to_uv
from sisinf.types import Candidate, Instance
from sisinf.validate import validate_candidate


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
) -> list[Candidate]:
    """Expand reduced-row candidates with bounded pairwise combinations."""

    if not inst.homogeneous:
        raise ValueError(f"{inst.name} is inhomogeneous; homogeneous candidate-pool search requires homogeneous input")
    if base_top_k < 0:
        raise ValueError(f"base_top_k must be non-negative, got {base_top_k}")

    singles = [np.asarray(vec, dtype=np.int64).reshape(-1) for vec in base_vecs[:base_top_k]]
    pairwise = generate_pairwise_combinations(
        singles,
        max_base=pair_max_base,
        pair_budget=pair_budget,
        include_negations=True,
    )
    vecs = dedup_integer_vectors([*singles, *pairwise])
    cands = decode_and_validate_vectors(inst, vecs)
    return rank_candidates(dedup_candidates(cands))


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
    ]
    for idx, cand in enumerate(cands[: min(total, preview)], start=1):
        max_linf = max(cand.linf_u, cand.linf_v)
        lines.append(
            "candidate[{idx}]: linf_u={linf_u}, linf_v={linf_v}, "
            "l2sq={l2sq}, max_linf={max_linf}, valid_main={valid_main}, "
            "valid_extra={valid_extra}".format(
                idx=idx,
                linf_u=cand.linf_u,
                linf_v=cand.linf_v,
                l2sq=cand.l2sq,
                max_linf=max_linf,
                valid_main=cand.valid_main,
                valid_extra=cand.valid_extra,
            )
        )
    return "\n".join(lines)
