"""Stage-4 scaffold for Wang 2025 Sieve-Then-Slice."""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass

import numpy as np

from sisinf.flexible_d4f import summarize_vector_lengths
from sisinf.metrics import l2sq_int
from sisinf.two_step import ShortVectorListBackend, summarize_short_vector_list
from sisinf.types import Instance

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SieveThenSliceProjectedSubLattice:
    """Working representation for one projected sublattice slice."""

    basis_rows: np.ndarray
    start: int
    stop: int
    ambient_dimension: int
    representation: str


@dataclass(frozen=True)
class SieveThenSliceResult:
    """Result object returned by the Stage-4 scaffold."""

    vectors: list[np.ndarray]
    kappa: int
    phi: int
    target_size: int
    base_sublattice: SieveThenSliceProjectedSubLattice
    upper_sublattice: SieveThenSliceProjectedSubLattice
    base_sieve_backend_name: str
    upper_sieve_backend_name: str
    backend_diagnostic_only: bool
    base_list_count: int
    upper_list_count: int
    lifted_t_count: int
    slicer_output_count: int
    final_candidate_count: int
    notes: tuple[str, ...] = ()


def estimate_plain_sieving_list_size(kappa: int) -> float:
    """Return the Wang-style baseline sieving list scale ``(sqrt(4/3))^kappa``."""

    if kappa < 0:
        raise ValueError(f"kappa must be non-negative, got {kappa}")
    return math.sqrt(4.0 / 3.0) ** kappa


def compute_sieve_then_slice_phi(target_size: int, kappa: int, oversampling_constant: int = 1) -> int:
    """Compute an engineering scaffold for the Wang 2025 ``phi`` parameter.

    This uses the paper's growth intuition that larger target list sizes require
    larger upper slices, but keeps the formula simple and robust for current
    engineering use.
    """

    if target_size < 1:
        raise ValueError(f"target_size must be >= 1, got {target_size}")
    if kappa < 0:
        raise ValueError(f"kappa must be non-negative, got {kappa}")
    if oversampling_constant < 0:
        raise ValueError(f"oversampling_constant must be non-negative, got {oversampling_constant}")

    baseline = estimate_plain_sieving_list_size(kappa)
    if target_size <= baseline:
        return 0
    growth_base = math.sqrt(4.0 / 3.0)
    phi = math.ceil(math.log(float(target_size) / baseline, growth_base)) + oversampling_constant
    return max(0, int(phi))


def extract_sieve_then_slice_projected_sublattice(
    B_row_red: np.ndarray,
    start: int,
    stop: int,
) -> SieveThenSliceProjectedSubLattice:
    """Extract the Stage-4 scaffold representation for ``L[start:stop]``."""

    arr = np.asarray(B_row_red, dtype=np.int64)
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
        raise ValueError(f"B_row_red must be a square two-dimensional matrix; got shape {arr.shape}")
    dim = arr.shape[0]
    if start < 0 or stop < start or stop > dim:
        raise ValueError(f"slice must satisfy 0 <= start <= stop <= {dim}, got start={start}, stop={stop}")
    return SieveThenSliceProjectedSubLattice(
        basis_rows=arr[start:stop, :].copy(),
        start=start,
        stop=stop,
        ambient_dimension=dim,
        representation="reduced_row_slice_scaffold_for_sieve_then_slice",
    )


def lift_upper_sublattice_vector_identity_scaffold(
    vec: np.ndarray,
    upper_sublattice: SieveThenSliceProjectedSubLattice,
) -> np.ndarray:
    """Return an engineering scaffold for ``Lift(T')``.

    Current diagnostic backends already emit ambient-coordinate vectors, so the
    lift is modeled as an identity map for the scaffold.
    """

    del upper_sublattice
    return np.asarray(vec, dtype=np.int64).reshape(-1).copy()


def modified_randomized_slicer_scaffold(
    T: list[np.ndarray],
    Lsieve: list[np.ndarray],
    target_size: int,
) -> list[np.ndarray]:
    """Engineering scaffold for the paper's modified randomized slicer.

    Exact paper logic is not implemented here. This scaffold deterministically
    combines ambient-coordinate vectors from ``T`` and ``Lsieve`` by subtraction
    to mimic the shape ``t - w`` of a slicing-style output, deduplicates exact
    repeats, and truncates to ``target_size``.
    """

    if target_size < 0:
        raise ValueError(f"target_size must be non-negative, got {target_size}")
    if target_size == 0:
        return []
    if not T or not Lsieve:
        return []

    out: list[np.ndarray] = []
    seen: set[tuple[int, ...]] = set()
    for t in T:
        t_arr = np.asarray(t, dtype=np.int64).reshape(-1)
        for w in Lsieve:
            w_arr = np.asarray(w, dtype=np.int64).reshape(-1)
            combo = t_arr - w_arr
            key = tuple(int(v) for v in combo)
            if key in seen:
                continue
            seen.add(key)
            out.append(combo.astype(np.int64, copy=True))
            if len(out) >= target_size:
                return out
    return out


def run_sieve_then_slice_on_reduced_basis(
    inst: Instance,
    B_row_red: np.ndarray,
    kappa: int,
    target_size: int,
    base_backend: ShortVectorListBackend,
    upper_backend: ShortVectorListBackend,
    oversampling_constant: int = 1,
) -> SieveThenSliceResult:
    """Run the Stage-4 Sieve-Then-Slice engineering scaffold on a reduced basis."""

    if not inst.homogeneous:
        raise ValueError(f"{inst.name} is inhomogeneous; Stage-4 currently supports homogeneous SIS∞ only")
    if base_backend is None or upper_backend is None:
        raise ValueError("both base_backend and upper_backend must be provided explicitly")

    arr = np.asarray(B_row_red, dtype=np.int64)
    dim = arr.shape[0]
    if kappa <= 0 or kappa > dim:
        raise ValueError(f"kappa must satisfy 1 <= kappa <= {dim}, got {kappa}")

    phi = compute_sieve_then_slice_phi(target_size=target_size, kappa=kappa, oversampling_constant=oversampling_constant)
    upper_stop = min(dim, kappa + phi)
    base_sublattice = extract_sieve_then_slice_projected_sublattice(arr, start=0, stop=kappa)
    upper_sublattice = extract_sieve_then_slice_projected_sublattice(arr, start=kappa, stop=upper_stop)

    logger.info("sieve_then_slice: kappa=%s", kappa)
    logger.info("sieve_then_slice: phi=%s", phi)
    logger.info("sieve_then_slice: target_size=%s", target_size)
    logger.info(
        "sieve_then_slice: base_sieve_backend_name=%s, upper_sieve_backend_name=%s",
        base_backend.name,
        upper_backend.name,
    )
    logger.info(
        "sieve_then_slice: backend_diagnostic_only=%s",
        bool(base_backend.diagnostic_only or upper_backend.diagnostic_only),
    )
    if base_backend.diagnostic_only or upper_backend.diagnostic_only:
        logger.warning(
            "sieve_then_slice: diagnostic-only backend in use; current output does not reproduce the Wang 2025 sieve lists or slicer"
        )

    base_list = base_backend.generate_short_vector_list(inst, base_sublattice)
    upper_list = upper_backend.generate_short_vector_list(inst, upper_sublattice)
    lifted_T = [lift_upper_sublattice_vector_identity_scaffold(vec, upper_sublattice) for vec in upper_list]
    slicer_output = modified_randomized_slicer_scaffold(lifted_T, base_list, target_size=target_size)

    final_vecs = [*base_list]
    seen = {tuple(int(v) for v in np.asarray(vec, dtype=np.int64).reshape(-1)) for vec in final_vecs}
    for vec in slicer_output:
        key = tuple(int(v) for v in np.asarray(vec, dtype=np.int64).reshape(-1))
        if key in seen:
            continue
        seen.add(key)
        final_vecs.append(np.asarray(vec, dtype=np.int64).reshape(-1).copy())

    logger.info("sieve_then_slice: base_list_count=%s", len(base_list))
    logger.info("sieve_then_slice: upper_list_count=%s", len(upper_list))
    logger.info("sieve_then_slice: lifted_T_count=%s", len(lifted_T))
    logger.info("sieve_then_slice: slicer_output_count=%s", len(slicer_output))
    logger.info("sieve_then_slice: final_candidate_count=%s", len(final_vecs))
    logger.info("sieve_then_slice: base_length_summary=%s", summarize_short_vector_list(base_list).replace("\n", "; "))
    logger.info("sieve_then_slice: upper_length_summary=%s", summarize_short_vector_list(upper_list).replace("\n", "; "))
    logger.info("sieve_then_slice: final_length_summary=%s", summarize_vector_lengths(final_vecs).replace("\n", "; "))

    notes = [
        "phi computation is an engineering scaffold inspired by Wang 2025 growth behavior",
        "Lift(T') is currently an identity scaffold in ambient coordinates",
        "modified_randomized_slicer_scaffold is a heuristic approximation, not the exact paper algorithm",
    ]
    if base_backend.diagnostic_only or upper_backend.diagnostic_only:
        notes.append("diagnostic-only backend in use")
        notes.append("current output is not a faithful Wang 2025 Sieve-Then-Slice list")

    return SieveThenSliceResult(
        vectors=final_vecs,
        kappa=kappa,
        phi=phi,
        target_size=target_size,
        base_sublattice=base_sublattice,
        upper_sublattice=upper_sublattice,
        base_sieve_backend_name=base_backend.name,
        upper_sieve_backend_name=upper_backend.name,
        backend_diagnostic_only=bool(base_backend.diagnostic_only or upper_backend.diagnostic_only),
        base_list_count=len(base_list),
        upper_list_count=len(upper_list),
        lifted_t_count=len(lifted_T),
        slicer_output_count=len(slicer_output),
        final_candidate_count=len(final_vecs),
        notes=tuple(notes),
    )
