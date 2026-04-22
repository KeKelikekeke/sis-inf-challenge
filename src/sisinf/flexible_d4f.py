"""Stage-3 scaffold for Wang 2025 FlexibleD4F."""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass

import numpy as np

from sisinf.metrics import l2sq_int, linf_norm_int
from sisinf.two_step import ShortVectorListBackend, TwoStepReductionTarget, summarize_short_vector_list
from sisinf.types import Instance

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class FlexibleD4FProjectedSubLattice:
    """Working representation of the Stage-3 projected sublattice ``L[f':kappa]``.

    The exact paper logic works with a projected sublattice. The current
    engineering scaffold represents that object by slicing the reduced row basis
    rows ``[f_prime:kappa]`` while keeping vectors in ambient coordinates.
    """

    basis_rows: np.ndarray
    f_prime: int
    kappa: int
    ambient_dimension: int
    representation: str


@dataclass(frozen=True)
class FlexibleD4FResult:
    """Result object returned by the Stage-3 FlexibleD4F scaffold."""

    vectors: list[np.ndarray]
    reduction_target: TwoStepReductionTarget
    projected_sublattice: FlexibleD4FProjectedSubLattice
    gamma_factor: float
    length_threshold: float
    backend_name: str
    backend_diagnostic_only: bool
    candidate_count_before_lift: int
    candidate_count_after_lift: int
    candidate_count_after_length_filter: int
    notes: tuple[str, ...] = ()


def flexible_d4f_gamma_factor(target_rhf: float, ambient_dimension: int, f_prime: int) -> float:
    """Return the Stage-3 gamma factor from Wang 2025 Section 5.1.

    This follows the paper formula
    ``gamma = sqrt(4/3) * delta^(-f' * d / (d - 1))``.
    """

    if target_rhf <= 0.0:
        raise ValueError(f"target_rhf must be positive, got {target_rhf}")
    if ambient_dimension <= 1:
        raise ValueError(f"ambient_dimension must be >= 2, got {ambient_dimension}")
    if f_prime < 0:
        raise ValueError(f"f_prime must be non-negative, got {f_prime}")
    exponent = -(f_prime * ambient_dimension) / (ambient_dimension - 1)
    return math.sqrt(4.0 / 3.0) * (float(target_rhf) ** exponent)


def extract_flexible_d4f_projected_sublattice(
    B_row_red: np.ndarray,
    kappa: int,
    f_prime: int,
) -> FlexibleD4FProjectedSubLattice:
    """Extract the Stage-3 scaffold representation of ``L[f':kappa]``."""

    arr = np.asarray(B_row_red, dtype=np.int64)
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
        raise ValueError(f"B_row_red must be a square two-dimensional matrix; got shape {arr.shape}")
    dim = arr.shape[0]
    if kappa <= 0 or kappa > dim:
        raise ValueError(f"kappa must satisfy 1 <= kappa <= {dim}, got {kappa}")
    if f_prime < 0 or f_prime > kappa:
        raise ValueError(f"f_prime must satisfy 0 <= f_prime <= kappa ({kappa}), got {f_prime}")
    return FlexibleD4FProjectedSubLattice(
        basis_rows=arr[f_prime:kappa, :].copy(),
        f_prime=f_prime,
        kappa=kappa,
        ambient_dimension=dim,
        representation="reduced_row_slice_scaffold_for_L[f_prime:kappa]",
    )


def estimate_gaussian_heuristic_from_row_basis(basis_rows: np.ndarray) -> float:
    """Estimate ``gh(L)`` from a row-basis matrix.

    This is a heuristic approximation used only by the Stage-3 scaffold. For a
    rectangular ``k x d`` row basis with full row rank, we estimate the lattice
    determinant by ``sqrt(det(B B^T))`` and then apply the standard Gaussian
    heuristic formula in dimension ``k``.
    """

    rows = np.asarray(basis_rows, dtype=np.float64)
    if rows.ndim != 2:
        raise ValueError(f"basis_rows must be two-dimensional; got shape {rows.shape}")
    k = rows.shape[0]
    if k <= 0:
        raise ValueError("basis_rows must contain at least one row")
    gram = rows @ rows.T
    det_gram = float(np.linalg.det(gram))
    if det_gram < 0.0:
        det_gram = 0.0
    det_lattice = math.sqrt(det_gram)
    unit_ball_volume = math.pi ** (k / 2.0) / math.gamma(k / 2.0 + 1.0)
    if unit_ball_volume <= 0.0:
        raise ValueError(f"invalid unit-ball volume for dimension {k}")
    return (det_lattice / unit_ball_volume) ** (1.0 / k)


def babai_lift_identity_scaffold(vec: np.ndarray, projected: FlexibleD4FProjectedSubLattice) -> np.ndarray:
    """Return an engineering scaffold for the Babai lift step.

    In the paper, vectors produced on ``L[f':kappa]`` are lifted back to the
    original lattice using Babai's nearest-plane style logic. The current
    diagnostic backends already emit vectors in ambient coordinates, so this
    Stage-3 scaffold models the lift as an identity map on those coordinates.
    """

    del projected
    return np.asarray(vec, dtype=np.int64).reshape(-1).copy()


def summarize_vector_lengths(vecs: list[np.ndarray]) -> str:
    """Return a compact ambient-vector length summary."""

    if not vecs:
        return "\n".join(
            [
                "lifted_vector_count=0",
                "lifted_vector_min_l2sq=NA",
                "lifted_vector_max_l2sq=NA",
                "lifted_vector_min_linf=NA",
                "lifted_vector_max_linf=NA",
            ]
        )
    l2s = [l2sq_int(np.asarray(vec, dtype=np.int64).reshape(-1)) for vec in vecs]
    linfs = [linf_norm_int(np.asarray(vec, dtype=np.int64).reshape(-1)) for vec in vecs]
    return "\n".join(
        [
            f"lifted_vector_count={len(vecs)}",
            f"lifted_vector_min_l2sq={min(l2s)}",
            f"lifted_vector_max_l2sq={max(l2s)}",
            f"lifted_vector_min_linf={min(linfs)}",
            f"lifted_vector_max_linf={max(linfs)}",
        ]
    )


def run_flexible_d4f_on_reduced_basis(
    inst: Instance,
    B_row_red: np.ndarray,
    reduction_target: TwoStepReductionTarget,
    kappa: int,
    f_prime: int,
    backend: ShortVectorListBackend,
) -> FlexibleD4FResult:
    """Run the Stage-3 FlexibleD4F engineering scaffold on a reduced basis."""

    if not inst.homogeneous:
        raise ValueError(f"{inst.name} is inhomogeneous; Stage-3 currently supports homogeneous SIS∞ only")
    if backend is None:
        raise ValueError("backend must be provided explicitly; no default backend is used")
    if reduction_target.target_rhf is None:
        raise NotImplementedError(
            "FlexibleD4F gamma computation currently requires an explicit target_rhf; beta-only reduction targets are not enough yet"
        )

    arr = np.asarray(B_row_red, dtype=np.int64)
    ambient_dimension = arr.shape[0]
    gamma_factor = flexible_d4f_gamma_factor(
        target_rhf=reduction_target.target_rhf,
        ambient_dimension=ambient_dimension,
        f_prime=f_prime,
    )
    base_sublattice = extract_flexible_d4f_projected_sublattice(arr, kappa=kappa, f_prime=0)
    base_gh = estimate_gaussian_heuristic_from_row_basis(base_sublattice.basis_rows)
    length_threshold = gamma_factor * base_gh
    projected = extract_flexible_d4f_projected_sublattice(arr, kappa=kappa, f_prime=f_prime)

    logger.info("flexible_d4f: reduction_target=%s", reduction_target.describe())
    logger.info("flexible_d4f: kappa=%s", kappa)
    logger.info("flexible_d4f: f_prime=%s", f_prime)
    logger.info("flexible_d4f: gamma_factor=%s", gamma_factor)
    logger.info("flexible_d4f: length_threshold=%s", length_threshold)
    logger.info(
        "flexible_d4f: projected_sublattice_dimension=%s, ambient_dimension=%s, representation=%s",
        projected.basis_rows.shape[0],
        projected.ambient_dimension,
        projected.representation,
    )
    logger.info(
        "flexible_d4f: short_vector_backend=%s, diagnostic_only=%s",
        backend.name,
        backend.diagnostic_only,
    )
    if backend.diagnostic_only:
        logger.warning(
            "flexible_d4f: backend %s is diagnostic-only and does not reproduce the Wang 2025 sieve output on L[f':kappa]",
            backend.name,
        )

    candidates = backend.generate_short_vector_list(inst, projected)
    lifted = [babai_lift_identity_scaffold(vec, projected) for vec in candidates]
    threshold_sq = length_threshold * length_threshold
    filtered = [vec for vec in lifted if l2sq_int(vec) <= threshold_sq]

    logger.info("flexible_d4f: candidate_count_before_lift=%s", len(candidates))
    logger.info("flexible_d4f: candidate_count_after_lift=%s", len(lifted))
    logger.info("flexible_d4f: candidate_count_after_length_filter=%s", len(filtered))
    logger.info("flexible_d4f: pre_lift_short_vector_summary=%s", summarize_short_vector_list(candidates).replace("\n", "; "))
    logger.info("flexible_d4f: lifted_vector_summary=%s", summarize_vector_lengths(lifted).replace("\n", "; "))

    notes: list[str] = [
        "length_threshold uses a Gaussian-heuristic estimate on L[0:kappa]",
        "Babai lift is currently an engineering identity scaffold in ambient coordinates",
    ]
    if backend.diagnostic_only:
        notes.append("diagnostic-only backend in use")
        notes.append("current output is not a faithful Wang 2025 FlexibleD4F vector list")
    return FlexibleD4FResult(
        vectors=filtered,
        reduction_target=reduction_target,
        projected_sublattice=projected,
        gamma_factor=gamma_factor,
        length_threshold=length_threshold,
        backend_name=backend.name,
        backend_diagnostic_only=backend.diagnostic_only,
        candidate_count_before_lift=len(candidates),
        candidate_count_after_lift=len(lifted),
        candidate_count_after_length_filter=len(filtered),
        notes=tuple(notes),
    )
