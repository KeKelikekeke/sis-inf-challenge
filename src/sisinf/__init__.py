"""Utilities for the 2026 SIS infinity-norm challenge engineering layer."""

from sisinf.io import load_problem, parse_problem_file
from sisinf.lattice import build_homogeneous_sis_basis_matrix
from sisinf.solver_hom_bkz import solve_homogeneous_bkz_baseline, solve_homogeneous_bkz_with_search
from sisinf.types import Candidate, Instance
from sisinf.validate import validate_candidate

__all__ = [
    "Candidate",
    "Instance",
    "build_homogeneous_sis_basis_matrix",
    "load_problem",
    "parse_problem_file",
    "solve_homogeneous_bkz_baseline",
    "solve_homogeneous_bkz_with_search",
    "validate_candidate",
]
