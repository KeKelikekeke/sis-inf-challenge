"""Probability helpers for Wang 2025 restricted-SVP modeling."""

from __future__ import annotations

import math


def standard_normal_cdf(x: float) -> float:
    """Return the standard normal CDF ``Phi(x)``."""

    return 0.5 * (1.0 + math.erf(float(x) / math.sqrt(2.0)))


def prob_infinity_norm_pass(dim: int, bound: float, len_bound: float) -> float:
    """Approximate ``P_B(len)`` for infinity-norm restricted SVP.

    This follows Wang 2025, Section 6.2:

    ``P_B(len) ~= (1 - 2 * Phi(-sqrt(dim) * bound / len)) ** dim``.

    The paper derives this from a spherical-Gaussian heuristic for vectors
    sampled uniformly from a Euclidean ball, so this function is explicitly a
    heuristic approximation rather than an exact probability law.
    """

    if dim <= 0:
        raise ValueError(f"dim must be positive, got {dim}")
    if bound < 0:
        raise ValueError(f"bound must be non-negative, got {bound}")
    if len_bound <= 0:
        raise ValueError(f"len_bound must be positive, got {len_bound}")

    ratio = math.sqrt(float(dim)) * float(bound) / float(len_bound)
    per_coordinate = 1.0 - 2.0 * standard_normal_cdf(-ratio)
    per_coordinate = min(1.0, max(0.0, per_coordinate))
    prob = per_coordinate ** dim
    return min(1.0, max(0.0, prob))


def required_list_size(p_success: float, p_single: float) -> float:
    """Return the target list size for one-vector success probability ``p_single``.

    Uses the Wang 2025 size relation:

    ``size = log(1 - p_success) / log(1 - p_single)``.

    Edge cases are normalized to practical values:
    - ``p_success == 0`` returns ``0.0``
    - ``p_single == 0`` returns ``math.inf``
    - ``p_single == 1`` returns ``1.0`` for any positive target success rate
    """

    if not 0.0 <= p_success < 1.0:
        raise ValueError(f"p_success must satisfy 0 <= p_success < 1, got {p_success}")
    if not 0.0 <= p_single <= 1.0:
        raise ValueError(f"p_single must satisfy 0 <= p_single <= 1, got {p_single}")
    if p_success == 0.0:
        return 0.0
    if p_single == 0.0:
        return math.inf
    if p_single == 1.0:
        return 1.0
    return math.log1p(-p_success) / math.log1p(-p_single)
