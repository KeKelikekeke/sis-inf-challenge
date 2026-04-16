"""Core data models for SIS infinity-norm challenge instances."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class Instance:
    """A normalized SIS infinity-norm problem instance."""

    name: str
    index: int
    n: int
    m: int
    q: int
    gamma: int
    A: np.ndarray
    t: np.ndarray | None
    require_l2_ge_q: bool
    homogeneous: bool
    source_path: Path


@dataclass(frozen=True)
class Candidate:
    """Validation result for a candidate pair ``(u, v)``."""

    u: np.ndarray
    v: np.ndarray
    linf_u: int
    linf_v: int
    l2sq: int
    congruence_ok: bool
    valid_main: bool
    valid_extra: bool
    meta: dict[str, Any] = field(default_factory=dict)
