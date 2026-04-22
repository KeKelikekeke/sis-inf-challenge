"""Restricted-SVP modeling helpers for homogeneous SIS infinity norm."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from sisinf.metrics import linf_norm_int
from sisinf.probability import prob_infinity_norm_pass, required_list_size
from sisinf.types import Candidate, Instance


RestrictionPredicate = Callable[[Candidate], bool]


@dataclass(frozen=True)
class RestrictedSVPProblem:
    """A Stage-1 restricted-SVP model instance.

    This is a modeling object only. It does not implement the Wang 2025 solver
    pipeline yet.
    """

    name: str
    dimension: int
    bound: int
    predicate: RestrictionPredicate
    probability_fn: Callable[[float], float]
    source_instance: Instance
    heuristic_notes: tuple[str, ...] = ()

    def restriction_holds(self, cand: Candidate) -> bool:
        """Return whether a validated candidate satisfies the restriction."""

        return bool(self.predicate(cand))

    def related_probability(self, len_bound: float) -> float:
        """Return the related probability ``P(len)`` for the modeled problem."""

        return float(self.probability_fn(len_bound))

    def required_list_size(self, p_success: float, len_bound: float) -> float:
        """Return the Wang 2025 target list size for this modeled problem."""

        return required_list_size(p_success=p_success, p_single=self.related_probability(len_bound))


def restriction_infinity_norm(cand: Candidate, bound: int) -> bool:
    """Return ``1[||[u;v]||_inf <= bound]`` for one candidate."""

    if bound < 0:
        raise ValueError(f"bound must be non-negative, got {bound}")
    return max(cand.linf_u, cand.linf_v) <= bound


def restriction_infinity_norm_vector(x: np.ndarray, bound: int) -> bool:
    """Return ``1[||x||_inf <= bound]`` for a raw integer vector."""

    if bound < 0:
        raise ValueError(f"bound must be non-negative, got {bound}")
    return linf_norm_int(np.asarray(x, dtype=np.int64).reshape(-1)) <= bound


def make_homogeneous_sis_infinity_restricted_svp(inst: Instance) -> RestrictedSVPProblem:
    """Model a homogeneous SIS∞ instance as a restricted-SVP problem.

    The restriction is:
    ``R([u; v]) = 1[ ||[u; v]||_inf <= gamma ]``

    For the current homogeneous SIS modeling, this is equivalent to requiring
    both ``||u||_inf <= gamma`` and ``||v||_inf <= gamma``.
    """

    if not inst.homogeneous:
        raise ValueError(f"{inst.name} is inhomogeneous; Stage 1 models homogeneous SIS∞ only")

    dim = inst.n + inst.m
    bound = int(inst.gamma)

    return RestrictedSVPProblem(
        name=f"{inst.name}_restricted_svp_linf",
        dimension=dim,
        bound=bound,
        predicate=lambda cand: restriction_infinity_norm(cand, bound),
        probability_fn=lambda len_bound: prob_infinity_norm_pass(dim=dim, bound=bound, len_bound=len_bound),
        source_instance=inst,
        heuristic_notes=(
            "related_probability uses the Wang 2025 Section 6.2 spherical-Gaussian heuristic approximation",
        ),
    )
