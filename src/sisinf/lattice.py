"""Lattice construction helpers for homogeneous SIS infinity-norm instances."""

from __future__ import annotations

from typing import Any

import numpy as np

from sisinf.types import Instance


def _require_homogeneous(inst: Instance) -> None:
    """Raise a clear error when a homogeneous-only function gets another instance."""

    if not inst.homogeneous:
        raise ValueError(f"{inst.name} is inhomogeneous; homogeneous SIS lattice baseline only accepts homogeneous instances")


def build_homogeneous_sis_basis_matrix(inst: Instance) -> np.ndarray:
    """Build the mathematical column-basis matrix ``[[qI, A], [0, -I]]``.

    For homogeneous SIS, a solution satisfies ``A v + u = q w``. With this
    basis, ``B * [w; -v] = [q w - A v; v] = [u; v]``. The lattice vectors
    therefore encode candidates directly as ``[u; v]``.
    """

    _require_homogeneous(inst)
    A = np.asarray(inst.A, dtype=np.int64)
    if A.shape != (inst.n, inst.m):
        raise ValueError(f"Instance A has shape {A.shape}; expected ({inst.n}, {inst.m}) for {inst.name}")

    dim = inst.n + inst.m
    B = np.zeros((dim, dim), dtype=np.int64)
    B[: inst.n, : inst.n] = inst.q * np.eye(inst.n, dtype=np.int64)
    B[: inst.n, inst.n :] = A
    B[inst.n :, inst.n :] = -np.eye(inst.m, dtype=np.int64)
    if B.shape != (dim, dim):
        raise ValueError(f"Internal basis construction error: got {B.shape}, expected ({dim}, {dim})")
    return B


def build_homogeneous_sis_row_basis(inst: Instance) -> np.ndarray:
    """Build the fpylll-friendly row-basis matrix for homogeneous SIS.

    The mathematical convention above treats ``B`` as a column-basis. fpylll
    reduces row bases, so this function returns ``B.T``. After row reduction,
    each row is still a vector in the same lattice and can be decoded directly
    as ``[u; v]`` without changing signs.
    """

    B = build_homogeneous_sis_basis_matrix(inst)
    return B.T.copy()


def to_fpylll_integer_matrix(B_row: np.ndarray) -> Any:
    """Convert a NumPy integer row-basis to ``fpylll.IntegerMatrix``.

    Raises:
        ImportError: If fpylll is not installed. Install it with a suitable
            environment command such as ``python -m pip install fpylll`` or
            ``conda install -c conda-forge fpylll``.
    """

    try:
        from fpylll import IntegerMatrix
    except ImportError as exc:
        raise ImportError(
            "BKZ/LLL functionality requires optional dependency fpylll. "
            "Install it with 'python -m pip install fpylll' or "
            "'conda install -c conda-forge fpylll'."
        ) from exc

    arr = np.asarray(B_row, dtype=np.int64)
    if arr.ndim != 2:
        raise ValueError(f"B_row must be two-dimensional; got shape {arr.shape}")
    rows, cols = arr.shape
    mat = IntegerMatrix(rows, cols)
    for i in range(rows):
        for j in range(cols):
            mat[i, j] = int(arr[i, j])
    return mat


def integer_matrix_to_numpy(M: Any) -> np.ndarray:
    """Convert an ``fpylll.IntegerMatrix``-like object to ``np.int64``."""

    rows = int(M.nrows)
    cols = int(M.ncols)
    arr = np.zeros((rows, cols), dtype=np.int64)
    for i in range(rows):
        for j in range(cols):
            arr[i, j] = int(M[i, j])
    return arr


def decode_lattice_vector_to_uv(x: np.ndarray, inst: Instance) -> tuple[np.ndarray, np.ndarray]:
    """Decode a lattice vector ``x = [u; v]`` into candidate arrays.

    Under the fixed homogeneous basis ``[[qI, A], [0, -I]]``, any coefficient
    vector ``[w; -v]`` maps to ``[q w - A v; v]``. The top block is ``u`` and
    the bottom block is ``v``. No extra sign flip is applied.
    """

    arr = np.asarray(x, dtype=np.int64).reshape(-1)
    expected = inst.n + inst.m
    if arr.shape != (expected,):
        raise ValueError(f"Lattice vector has shape {arr.shape}; expected ({expected},) for {inst.name}")
    u = arr[: inst.n].copy()
    v = arr[inst.n :].copy()
    return u, v


def extract_row_vectors(B_row_red: np.ndarray, limit: int | None = None) -> list[np.ndarray]:
    """Extract the first ``limit`` rows from a reduced row-basis matrix."""

    arr = np.asarray(B_row_red, dtype=np.int64)
    if arr.ndim != 2:
        raise ValueError(f"B_row_red must be two-dimensional; got shape {arr.shape}")
    row_count = arr.shape[0] if limit is None else min(limit, arr.shape[0])
    if row_count < 0:
        raise ValueError(f"limit must be non-negative or None, got {limit}")
    return [arr[i, :].copy() for i in range(row_count)]
