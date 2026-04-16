"""Integer metrics used by SIS infinity-norm validation."""

from __future__ import annotations

import numpy as np


def linf_norm_int(x: np.ndarray) -> int:
    """Return the integer infinity norm of a vector-like array."""

    arr = np.asarray(x, dtype=np.int64)
    if arr.size == 0:
        return 0
    return int(np.max(np.abs(arr)))


def l2sq_int(x: np.ndarray) -> int:
    """Return the squared Euclidean norm using integer arithmetic."""

    arr = np.asarray(x, dtype=np.int64).reshape(-1)
    return int(arr @ arr)


def center_mod_q(x: np.ndarray, q: int) -> np.ndarray:
    """Map residues modulo ``q`` to a symmetric integer representative range."""

    if q <= 0:
        raise ValueError(f"Modulus q must be positive, got {q}")
    arr = np.asarray(x, dtype=np.int64)
    return ((arr + q // 2) % q - q // 2).astype(np.int64, copy=False)
