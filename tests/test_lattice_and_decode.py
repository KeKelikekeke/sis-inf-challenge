from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sisinf.lattice import (  # noqa: E402
    build_homogeneous_sis_basis_matrix,
    build_homogeneous_sis_row_basis,
    decode_lattice_vector_to_uv,
    extract_row_vectors,
)
from sisinf.types import Instance  # noqa: E402


def _small_inst(homogeneous: bool = True) -> Instance:
    """Build a tiny artificial instance for lattice tests."""

    A = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int64)
    t = np.zeros(2, dtype=np.int64)
    return Instance(
        name="small",
        index=0,
        n=2,
        m=3,
        q=7,
        gamma=3,
        A=A,
        t=t,
        require_l2_ge_q=False,
        homogeneous=homogeneous,
        source_path=Path("memory"),
    )


def test_build_homogeneous_sis_basis_matrix_blocks() -> None:
    """The mathematical basis has blocks [[qI, A], [0, -I]]."""

    inst = _small_inst()
    B = build_homogeneous_sis_basis_matrix(inst)
    assert B.shape == (5, 5)
    np.testing.assert_array_equal(B[:2, :2], 7 * np.eye(2, dtype=np.int64))
    np.testing.assert_array_equal(B[:2, 2:], inst.A)
    np.testing.assert_array_equal(B[2:, :2], np.zeros((3, 2), dtype=np.int64))
    np.testing.assert_array_equal(B[2:, 2:], -np.eye(3, dtype=np.int64))


def test_build_homogeneous_sis_row_basis_is_transpose() -> None:
    """The fpylll row-basis follows the documented transpose convention."""

    inst = _small_inst()
    B = build_homogeneous_sis_basis_matrix(inst)
    B_row = build_homogeneous_sis_row_basis(inst)
    assert B_row.shape == (5, 5)
    np.testing.assert_array_equal(B_row, B.T)


def test_decode_lattice_vector_to_uv_preserves_sign_convention() -> None:
    """A vector [q w - A v; v] decodes to u, v and satisfies A v + u == 0 mod q."""

    inst = _small_inst()
    w = np.array([1, -1], dtype=np.int64)
    v = np.array([2, -1, 3], dtype=np.int64)
    u = inst.q * w - inst.A @ v
    x = np.concatenate([u, v])
    decoded_u, decoded_v = decode_lattice_vector_to_uv(x, inst)
    np.testing.assert_array_equal(decoded_u, u)
    np.testing.assert_array_equal(decoded_v, v)
    np.testing.assert_array_equal((inst.A @ decoded_v + decoded_u) % inst.q, np.zeros(inst.n, dtype=np.int64))


def test_invalid_inputs_raise() -> None:
    """Non-homogeneous instances and bad vector dimensions raise clear errors."""

    with pytest.raises(ValueError, match="inhomogeneous"):
        build_homogeneous_sis_basis_matrix(_small_inst(homogeneous=False))
    with pytest.raises(ValueError, match="expected"):
        decode_lattice_vector_to_uv(np.zeros(4, dtype=np.int64), _small_inst())
    with pytest.raises(ValueError, match="two-dimensional"):
        extract_row_vectors(np.zeros(3, dtype=np.int64))
