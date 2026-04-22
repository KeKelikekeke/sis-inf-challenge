"""Stage-5 scaffold for Wang 2025 Algorithm 8 on homogeneous SIS infinity norm."""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass

import numpy as np

from sisinf.flexible_d4f import (
    estimate_gaussian_heuristic_from_row_basis,
    flexible_d4f_gamma_factor,
    run_flexible_d4f_on_reduced_basis,
    summarize_vector_lengths,
)
from sisinf.lattice import build_homogeneous_sis_row_basis, decode_lattice_vector_to_uv
from sisinf.restricted_svp import RestrictedSVPProblem, make_homogeneous_sis_infinity_restricted_svp
from sisinf.sieve_then_slice import estimate_plain_sieving_list_size, run_sieve_then_slice_on_reduced_basis
from sisinf.solver_hom_bkz import run_bkz_on_row_basis, run_lll_on_row_basis
from sisinf.two_step import DiagnosticReducedRowBackend, ShortVectorListBackend, TwoStepReductionTarget, summarize_required_list_size
from sisinf.types import Candidate, Instance
from sisinf.validate import validate_candidate

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RestrictedHomogeneousSolverResult:
    """Structured result for the Stage-5 homogeneous restricted-SVP dispatcher."""

    selected_branch: str
    len_bound: float
    single_vector_pass_probability: float
    raw_required_size: float
    integer_required_size: int | float
    threshold_size: float
    backend_name: str
    backend_diagnostic_only: bool
    candidate_count_before_restriction: int
    candidate_count_after_restriction: int
    restriction_pass_count: int
    restriction_pass_vectors: list[np.ndarray]
    notes: tuple[str, ...] = ()
    limitations: tuple[str, ...] = ()

    @property
    def produced_list_size(self) -> int:
        """Return the pre-restriction candidate list size."""

        return self.candidate_count_before_restriction


def estimate_row_gram_schmidt_norms(B_row_red: np.ndarray) -> list[float]:
    """Return floating-point Gram-Schmidt norms for the row basis."""

    rows = np.asarray(B_row_red, dtype=np.float64)
    if rows.ndim != 2:
        raise ValueError(f"B_row_red must be two-dimensional; got shape {rows.shape}")

    ortho: list[np.ndarray] = []
    norms: list[float] = []
    for row in rows:
        vec = row.copy()
        for base in ortho:
            denom = float(base @ base)
            if denom == 0.0:
                continue
            vec = vec - (float(vec @ base) / denom) * base
        ortho.append(vec)
        norms.append(float(np.linalg.norm(vec)))
    return norms


def compute_algorithm8_len_bound(B_row_red: np.ndarray, kappa: int) -> float:
    """Return the Stage-5 Algorithm 8 length bound.

    Exact paper logic:
    ``len = sqrt(4/3) * sqrt(gh(L(B[0:kappa]))^2 + ||b*_kappa||^2)``.

    Engineering simplification:
    current code estimates both ``gh(L(B[0:kappa]))`` and ``||b*_kappa||`` from
    the reduced row basis already available in the scaffold implementation.
    """

    arr = np.asarray(B_row_red, dtype=np.int64)
    dim = arr.shape[0]
    if kappa <= 0 or kappa > dim:
        raise ValueError(f"kappa must satisfy 1 <= kappa <= {dim}, got {kappa}")

    prefix_gh = estimate_gaussian_heuristic_from_row_basis(arr[:kappa, :])
    gs_norms = estimate_row_gram_schmidt_norms(arr)
    gs_index = min(kappa, dim - 1)
    tail_gs_norm = gs_norms[gs_index]
    return math.sqrt(4.0 / 3.0) * math.sqrt(prefix_gh * prefix_gh + tail_gs_norm * tail_gs_norm)


def select_algorithm8_branch(integer_required_size: int | float, kappa: int) -> tuple[str, float]:
    """Select the Stage-5 list-generation branch from Algorithm 8."""

    threshold_size = estimate_plain_sieving_list_size(kappa)
    if integer_required_size <= threshold_size:
        return "flexible_d4f", threshold_size
    return "sieve_then_slice", threshold_size


def choose_flexible_d4f_f_prime(
    restricted_problem: RestrictedSVPProblem,
    reduction_target: TwoStepReductionTarget,
    B_row_red: np.ndarray,
    kappa: int,
    p_success: float,
) -> tuple[int, float]:
    """Choose the Stage-5 FlexibleD4F ``f_prime`` following Algorithm 8."""

    if reduction_target.target_rhf is None:
        raise NotImplementedError(
            "Algorithm 8 FlexibleD4F dispatch currently requires target_rhf; beta-only Stage-5 tuning is not implemented"
        )

    arr = np.asarray(B_row_red, dtype=np.int64)
    ambient_dimension = arr.shape[0]
    base_gh = estimate_gaussian_heuristic_from_row_basis(arr[:kappa, :])
    current_f = 0
    current_len = math.sqrt(4.0 / 3.0) * base_gh
    current_size = estimate_plain_sieving_list_size(kappa)

    while current_f < kappa and current_size >= restricted_problem.required_list_size(p_success=p_success, len_bound=current_len):
        current_f += 1
        gamma = flexible_d4f_gamma_factor(
            target_rhf=reduction_target.target_rhf,
            ambient_dimension=ambient_dimension,
            f_prime=current_f,
        )
        current_len = gamma * base_gh
        current_size = gamma**kappa

    selected_f = max(0, current_f - 1)
    if selected_f == 0:
        return selected_f, math.sqrt(4.0 / 3.0) * base_gh
    gamma = flexible_d4f_gamma_factor(
        target_rhf=reduction_target.target_rhf,
        ambient_dimension=ambient_dimension,
        f_prime=selected_f,
    )
    return selected_f, gamma * base_gh


def _default_backend_for_required_size(required_size: int | float) -> DiagnosticReducedRowBackend:
    """Return the default diagnostic backend used by the Stage-5 scaffold."""

    if math.isinf(required_size):
        top_k = 50
    else:
        top_k = max(20, min(200, int(required_size)))
    return DiagnosticReducedRowBackend(top_k=top_k)


def _vector_to_candidate(inst: Instance, vec: np.ndarray) -> Candidate:
    u, v = decode_lattice_vector_to_uv(vec, inst)
    return validate_candidate(inst, u=u, v=v)


def _restriction_scan(
    inst: Instance,
    restricted_problem: RestrictedSVPProblem,
    vectors: list[np.ndarray],
) -> list[np.ndarray]:
    passed: list[np.ndarray] = []
    for vec in vectors:
        cand = _vector_to_candidate(inst, vec)
        if restricted_problem.restriction_holds(cand):
            passed.append(np.asarray(vec, dtype=np.int64).reshape(-1).copy())
    return passed


def run_restricted_svp_dispatcher_on_reduced_basis(
    inst: Instance,
    B_row_red: np.ndarray,
    reduction_target: TwoStepReductionTarget,
    kappa: int,
    p_success: float,
    restricted_problem: RestrictedSVPProblem | None = None,
    flexible_backend: ShortVectorListBackend | None = None,
    sieve_base_backend: ShortVectorListBackend | None = None,
    sieve_upper_backend: ShortVectorListBackend | None = None,
    oversampling_constant: int = 1,
) -> RestrictedHomogeneousSolverResult:
    """Run the Stage-5 Algorithm 8 dispatcher on an already reduced basis."""

    if not inst.homogeneous:
        raise ValueError(f"{inst.name} is inhomogeneous; Stage-5 currently supports homogeneous SIS infinity-norm only")

    model = restricted_problem or make_homogeneous_sis_infinity_restricted_svp(inst)
    len_bound = compute_algorithm8_len_bound(B_row_red, kappa=kappa)
    p_single = model.related_probability(len_bound)
    required_summary = summarize_required_list_size(p_success=p_success, p_single=p_single)
    selected_branch, threshold_size = select_algorithm8_branch(required_summary.integer_required_size, kappa=kappa)

    if flexible_backend is None:
        flexible_backend = _default_backend_for_required_size(required_summary.integer_required_size)
    if sieve_base_backend is None:
        sieve_base_backend = _default_backend_for_required_size(required_summary.integer_required_size)
    if sieve_upper_backend is None:
        sieve_upper_backend = _default_backend_for_required_size(required_summary.integer_required_size)

    logger.info("restricted_hom: reduction target beta=%s / target_rhf=%s", reduction_target.beta, reduction_target.target_rhf)
    logger.info("restricted_hom: kappa=%s", kappa)
    logger.info("restricted_hom: len_bound=%s", len_bound)
    logger.info("restricted_hom: single_vector_pass_probability=%s", p_single)
    logger.info("restricted_hom: raw_required_size=%s", required_summary.raw_required_size)
    logger.info("restricted_hom: integer_required_size=%s", required_summary.integer_required_size)
    logger.info("restricted_hom: threshold_size=%s", threshold_size)
    logger.info("restricted_hom: selected_branch=%s", selected_branch)

    produced_vectors: list[np.ndarray]
    backend_name: str
    backend_diagnostic_only: bool

    if selected_branch == "flexible_d4f":
        selected_f_prime, flexible_len_bound = choose_flexible_d4f_f_prime(
            model,
            reduction_target=reduction_target,
            B_row_red=B_row_red,
            kappa=kappa,
            p_success=p_success,
        )
        branch_result = run_flexible_d4f_on_reduced_basis(
            inst,
            B_row_red,
            reduction_target=reduction_target,
            kappa=kappa,
            f_prime=selected_f_prime,
            backend=flexible_backend,
        )
        produced_vectors = branch_result.vectors
        backend_name = branch_result.backend_name
        backend_diagnostic_only = branch_result.backend_diagnostic_only
        len_bound = flexible_len_bound
        logger.info("restricted_hom: flexible_d4f_f_prime=%s", selected_f_prime)
    else:
        branch_result = run_sieve_then_slice_on_reduced_basis(
            inst,
            B_row_red,
            kappa=kappa,
            target_size=int(required_summary.integer_required_size)
            if not math.isinf(required_summary.integer_required_size)
            else int(math.ceil(threshold_size)) + 1,
            base_backend=sieve_base_backend,
            upper_backend=sieve_upper_backend,
            oversampling_constant=oversampling_constant,
        )
        produced_vectors = branch_result.vectors
        backend_name = f"{branch_result.base_sieve_backend_name}+{branch_result.upper_sieve_backend_name}"
        backend_diagnostic_only = branch_result.backend_diagnostic_only

    passed_vectors = _restriction_scan(inst, model, produced_vectors)

    logger.info("restricted_hom: backend_name=%s", backend_name)
    logger.info("restricted_hom: backend_diagnostic_only=%s", backend_diagnostic_only)
    logger.info("restricted_hom: candidate_count_before_restriction=%s", len(produced_vectors))
    logger.info("restricted_hom: candidate_count_after_restriction=%s", len(passed_vectors))
    logger.info("restricted_hom: restriction_pass_count=%s", len(passed_vectors))
    logger.info("restricted_hom: final_candidate_length_summary=%s", summarize_vector_lengths(produced_vectors).replace("\n", "; "))

    notes = [
        "exact paper logic: branch selection follows Algorithm 8 threshold comparison against (sqrt(4/3))^kappa",
        "heuristic approximation: single_vector_pass_probability uses the Stage-1 spherical-Gaussian P(len) model",
        "engineering simplification: len_bound is estimated from the reduced row basis and row Gram-Schmidt norms",
    ]
    limitations: list[str] = []
    if selected_branch == "flexible_d4f":
        notes.append("engineering simplification: FlexibleD4F parameter tuning uses the Stage-3 scaffold loop on f_prime")
    else:
        notes.append("engineering simplification: Sieve-Then-Slice target_size uses integer_required_size for dispatch")
    if backend_diagnostic_only:
        limitations.append(
            "diagnostic-only backend limitation: current vector generation is scaffold-only and is not a faithful Wang 2025 sieve implementation"
        )
    limitations.append("current Stage-5 entry handles homogeneous SIS infinity-norm only")
    limitations.append("current Stage-5 entry does not implement Kannan embedding or inhomogeneous SIS infinity-norm")

    return RestrictedHomogeneousSolverResult(
        selected_branch=selected_branch,
        len_bound=len_bound,
        single_vector_pass_probability=p_single,
        raw_required_size=required_summary.raw_required_size,
        integer_required_size=required_summary.integer_required_size,
        threshold_size=threshold_size,
        backend_name=backend_name,
        backend_diagnostic_only=backend_diagnostic_only,
        candidate_count_before_restriction=len(produced_vectors),
        candidate_count_after_restriction=len(passed_vectors),
        restriction_pass_count=len(passed_vectors),
        restriction_pass_vectors=passed_vectors,
        notes=tuple(notes),
        limitations=tuple(limitations),
    )


def solve_homogeneous_restricted_svp(
    inst: Instance,
    reduction_target: TwoStepReductionTarget,
    kappa: int,
    p_success: float,
    restricted_problem: RestrictedSVPProblem | None = None,
    flexible_backend: ShortVectorListBackend | None = None,
    sieve_base_backend: ShortVectorListBackend | None = None,
    sieve_upper_backend: ShortVectorListBackend | None = None,
    oversampling_constant: int = 1,
) -> RestrictedHomogeneousSolverResult:
    """Solver-facing Stage-5 entry for homogeneous SIS infinity-norm restricted-SVP solving."""

    if not inst.homogeneous:
        raise ValueError(f"{inst.name} is inhomogeneous; Stage-5 currently supports homogeneous SIS infinity-norm only")
    if reduction_target.beta is None:
        raise NotImplementedError(
            "Stage-5 homogeneous entry currently requires an explicit BKZ beta; target_rhf-only reduction is not implemented"
        )

    B_row = build_homogeneous_sis_row_basis(inst)
    B_row_lll = run_lll_on_row_basis(B_row)
    B_row_bkz = run_bkz_on_row_basis(B_row_lll, beta=reduction_target.beta, max_loops=reduction_target.max_loops)
    return run_restricted_svp_dispatcher_on_reduced_basis(
        inst,
        B_row_bkz,
        reduction_target=reduction_target,
        kappa=kappa,
        p_success=p_success,
        restricted_problem=restricted_problem,
        flexible_backend=flexible_backend,
        sieve_base_backend=sieve_base_backend,
        sieve_upper_backend=sieve_upper_backend,
        oversampling_constant=oversampling_constant,
    )
