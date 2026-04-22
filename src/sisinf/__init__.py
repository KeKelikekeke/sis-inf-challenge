"""Utilities for the 2026 SIS infinity-norm challenge engineering layer."""

from sisinf.flexible_d4f import (
    FlexibleD4FProjectedSubLattice,
    FlexibleD4FResult,
    extract_flexible_d4f_projected_sublattice,
    flexible_d4f_gamma_factor,
    run_flexible_d4f_on_reduced_basis,
)
from sisinf.embedding import build_kannan_embedding_basis_matrix, solve_inhomogeneous_embedding_skeleton
from sisinf.io import load_problem, parse_problem_file
from sisinf.lattice import build_homogeneous_sis_basis_matrix
from sisinf.probability import prob_infinity_norm_pass, required_list_size
from sisinf.restricted_svp import RestrictedSVPProblem, make_homogeneous_sis_infinity_restricted_svp
from sisinf.sieve_then_slice import (
    SieveThenSliceProjectedSubLattice,
    SieveThenSliceResult,
    compute_sieve_then_slice_phi,
    extract_sieve_then_slice_projected_sublattice,
    lift_upper_sublattice_vector_identity_scaffold,
    modified_randomized_slicer_scaffold,
    run_sieve_then_slice_on_reduced_basis,
)
from sisinf.solver import select_solver_path, solve_instance_baseline
from sisinf.solver_hom_bkz import solve_homogeneous_bkz_baseline, solve_homogeneous_bkz_with_search
from sisinf.two_step import (
    DiagnosticReducedRowBackend,
    ProjectedSubLattice,
    RequiredListSizeSummary,
    ShortVectorListBackend,
    TwoStepReductionTarget,
    TwoStepSolverResult,
    extract_projected_sublattice,
    run_two_step_on_reduced_basis,
    solve_two_step_homogeneous,
    summarize_required_list_size,
)
from sisinf.types import Candidate, Instance
from sisinf.validate import validate_candidate

__all__ = [
    "Candidate",
    "Instance",
    "build_homogeneous_sis_basis_matrix",
    "build_kannan_embedding_basis_matrix",
    "FlexibleD4FProjectedSubLattice",
    "FlexibleD4FResult",
    "load_problem",
    "make_homogeneous_sis_infinity_restricted_svp",
    "parse_problem_file",
    "prob_infinity_norm_pass",
    "required_list_size",
    "RestrictedSVPProblem",
    "DiagnosticReducedRowBackend",
    "ProjectedSubLattice",
    "RequiredListSizeSummary",
    "SieveThenSliceProjectedSubLattice",
    "SieveThenSliceResult",
    "ShortVectorListBackend",
    "compute_sieve_then_slice_phi",
    "extract_sieve_then_slice_projected_sublattice",
    "lift_upper_sublattice_vector_identity_scaffold",
    "modified_randomized_slicer_scaffold",
    "select_solver_path",
    "solve_homogeneous_bkz_baseline",
    "solve_homogeneous_bkz_with_search",
    "solve_inhomogeneous_embedding_skeleton",
    "solve_instance_baseline",
    "run_flexible_d4f_on_reduced_basis",
    "run_sieve_then_slice_on_reduced_basis",
    "solve_two_step_homogeneous",
    "summarize_required_list_size",
    "TwoStepReductionTarget",
    "TwoStepSolverResult",
    "extract_flexible_d4f_projected_sublattice",
    "extract_projected_sublattice",
    "flexible_d4f_gamma_factor",
    "run_two_step_on_reduced_basis",
    "validate_candidate",
]
