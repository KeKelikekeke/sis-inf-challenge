"""Stage-2 scaffold for Wang 2025 two-step approximate-SVP solving."""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from sisinf.lattice import build_homogeneous_sis_row_basis, extract_row_vectors
from sisinf.metrics import l2sq_int, linf_norm_int
from sisinf.probability import required_list_size
from sisinf.solver_hom_bkz import run_bkz_on_row_basis, run_lll_on_row_basis
from sisinf.types import Instance

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TwoStepReductionTarget:
    """Reduction target for the Stage-2 two-step scaffold.

    The Wang 2025 interface is expressed in terms of basis quality / RHF
    ``delta``. The current engineering scaffold can still be driven by BKZ
    ``beta`` while keeping the distinction explicit in the interface.
    """

    beta: int | None = None
    target_rhf: float | None = None
    max_loops: int = 2

    def describe(self) -> str:
        """Return a compact human-readable description."""

        parts = []
        if self.beta is not None:
            parts.append(f"beta={self.beta}")
        if self.target_rhf is not None:
            parts.append(f"target_rhf={self.target_rhf}")
        parts.append(f"max_loops={self.max_loops}")
        return ", ".join(parts)


@dataclass(frozen=True)
class ProjectedSubLattice:
    """Working representation of the projected sublattice ``L[0:kappa]``.

    The exact sieve backend from Wang 2025 is not implemented yet. This Stage-2
    scaffold therefore stores the first ``kappa`` reduced basis rows as the
    current engineering representation consumed by pluggable short-vector
    backends.
    """

    basis_rows: np.ndarray
    kappa: int
    ambient_dimension: int
    representation: str


@dataclass(frozen=True)
class RequiredListSizeSummary:
    """Solver-facing wrapper around the Stage-1 required-list-size formula."""

    p_success: float
    p_single: float
    raw_required_size: float
    integer_required_size: int


@dataclass(frozen=True)
class TwoStepSolverResult:
    """Result object returned by the Stage-2 two-step scaffold."""

    vectors: list[np.ndarray]
    reduction_target: TwoStepReductionTarget
    projected_sublattice: ProjectedSubLattice
    backend_name: str
    backend_diagnostic_only: bool
    notes: tuple[str, ...] = ()


class ShortVectorListBackend(Protocol):
    """Interface for generating a short-vector list from ``L[0:kappa]``."""

    name: str
    diagnostic_only: bool

    def generate_short_vector_list(
        self,
        inst: Instance,
        projected: ProjectedSubLattice,
    ) -> list[np.ndarray]:
        """Return a list of short lattice vectors for the projected sublattice."""


@dataclass(frozen=True)
class DiagnosticReducedRowBackend:
    """Diagnostic-only reduced-row backend.

    This backend is an engineering placeholder. It returns the first few rows of
    the current projected representation and must not be interpreted as a real
    sieve backend or as a faithful reproduction of Wang 2025 TwoStepSolver.
    """

    top_k: int = 20
    name: str = "diagnostic_reduced_row_backend"
    diagnostic_only: bool = True

    def generate_short_vector_list(
        self,
        inst: Instance,
        projected: ProjectedSubLattice,
    ) -> list[np.ndarray]:
        del inst
        if self.top_k < 0:
            raise ValueError(f"top_k must be non-negative, got {self.top_k}")
        rows = extract_row_vectors(projected.basis_rows, limit=self.top_k)
        return [row.copy() for row in rows]


def summarize_required_list_size(p_success: float, p_single: float) -> RequiredListSizeSummary:
    """Return both raw and integerized required list sizes for solver-facing code."""

    raw = required_list_size(p_success=p_success, p_single=p_single)
    if math.isinf(raw):
        integer = math.inf
    else:
        integer = int(math.ceil(max(1.0, raw)))
    return RequiredListSizeSummary(
        p_success=p_success,
        p_single=p_single,
        raw_required_size=raw,
        integer_required_size=integer,
    )


def extract_projected_sublattice(B_row_red: np.ndarray, kappa: int) -> ProjectedSubLattice:
    """Extract the working representation for the projected sublattice ``L[0:kappa]``."""

    arr = np.asarray(B_row_red, dtype=np.int64)
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
        raise ValueError(f"B_row_red must be a square two-dimensional matrix; got shape {arr.shape}")
    dim = arr.shape[0]
    if kappa <= 0 or kappa > dim:
        raise ValueError(f"kappa must satisfy 1 <= kappa <= {dim}, got {kappa}")
    return ProjectedSubLattice(
        basis_rows=arr[:kappa, :].copy(),
        kappa=kappa,
        ambient_dimension=dim,
        representation="reduced_row_prefix_scaffold_for_projected_sublattice",
    )


def summarize_short_vector_list(vecs: list[np.ndarray]) -> str:
    """Format a compact summary of a generated short-vector list."""

    if not vecs:
        return "\n".join(
            [
                "short_vector_list_count=0",
                "short_vector_list_min_l2sq=NA",
                "short_vector_list_max_l2sq=NA",
                "short_vector_list_min_linf=NA",
                "short_vector_list_max_linf=NA",
            ]
        )
    l2s = [l2sq_int(np.asarray(vec, dtype=np.int64).reshape(-1)) for vec in vecs]
    linfs = [linf_norm_int(np.asarray(vec, dtype=np.int64).reshape(-1)) for vec in vecs]
    return "\n".join(
        [
            f"short_vector_list_count={len(vecs)}",
            f"short_vector_list_min_l2sq={min(l2s)}",
            f"short_vector_list_max_l2sq={max(l2s)}",
            f"short_vector_list_min_linf={min(linfs)}",
            f"short_vector_list_max_linf={max(linfs)}",
        ]
    )


def run_two_step_on_reduced_basis(
    inst: Instance,
    B_row_red: np.ndarray,
    reduction_target: TwoStepReductionTarget,
    kappa: int,
    backend: ShortVectorListBackend,
) -> TwoStepSolverResult:
    """Run the Stage-2 two-step scaffold on an already reduced row basis."""

    if backend is None:
        raise ValueError("backend must be provided explicitly; no default backend is used")

    projected = extract_projected_sublattice(B_row_red, kappa=kappa)
    logger.info("two_step: reduction_target=%s", reduction_target.describe())
    logger.info("two_step: kappa=%s", kappa)
    logger.info(
        "two_step: projected_sublattice_dimension=%s, ambient_dimension=%s, representation=%s",
        projected.kappa,
        projected.ambient_dimension,
        projected.representation,
    )
    logger.info(
        "two_step: short_vector_backend=%s, diagnostic_only=%s",
        backend.name,
        backend.diagnostic_only,
    )
    if backend.diagnostic_only:
        logger.warning(
            "two_step: backend %s is diagnostic-only and does not reproduce the Wang 2025 sieve output",
            backend.name,
        )
    vecs = backend.generate_short_vector_list(inst, projected)
    logger.info("two_step: generated_short_vector_count=%s", len(vecs))
    logger.info("two_step: %s", summarize_short_vector_list(vecs).replace("\n", "; "))
    notes: tuple[str, ...] = ()
    if backend.diagnostic_only:
        notes = (
            "diagnostic-only backend in use",
            "current output is not a faithful Wang 2025 sieve list",
        )
    return TwoStepSolverResult(
        vectors=vecs,
        reduction_target=reduction_target,
        projected_sublattice=projected,
        backend_name=backend.name,
        backend_diagnostic_only=backend.diagnostic_only,
        notes=notes,
    )


def solve_two_step_homogeneous(
    inst: Instance,
    reduction_target: TwoStepReductionTarget,
    kappa: int,
    backend: ShortVectorListBackend,
) -> TwoStepSolverResult:
    """Run the Stage-2 two-step scaffold on a homogeneous SIS lattice.

    This function keeps the reduction target interface explicit. At the current
    stage, only BKZ ``beta``-driven reduction is implemented. A pure RHF-driven
    backend is not implemented yet and is therefore surfaced as a missing gap
    instead of silently approximated.
    """

    if not inst.homogeneous:
        raise ValueError(f"{inst.name} is inhomogeneous; Stage-2 currently supports homogeneous SIS∞ only")
    if backend is None:
        raise ValueError("backend must be provided explicitly; no default backend is used")
    if reduction_target.beta is None:
        raise NotImplementedError(
            "Two-step scaffold currently requires an explicit BKZ beta; target_rhf-only reduction is not implemented yet"
        )

    B_row = build_homogeneous_sis_row_basis(inst)
    B_row_lll = run_lll_on_row_basis(B_row)
    B_row_bkz = run_bkz_on_row_basis(B_row_lll, beta=reduction_target.beta, max_loops=reduction_target.max_loops)
    return run_two_step_on_reduced_basis(
        inst,
        B_row_bkz,
        reduction_target=reduction_target,
        kappa=kappa,
        backend=backend,
    )
