from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sisinf.embedding import (  # noqa: E402
    build_kannan_embedding_basis_matrix,
    build_kannan_embedding_row_basis,
    decode_embedding_vector_to_uv,
    solve_inhomogeneous_embedding_skeleton,
    validate_embedding_vector_candidate,
)
from sisinf.solver import select_solver_path, solve_instance_baseline  # noqa: E402
from sisinf.solver_hom_bkz import collect_homogeneous_candidates_from_row_basis  # noqa: E402
from sisinf.types import Instance  # noqa: E402


def _inst(homogeneous: bool, t: np.ndarray) -> Instance:
    """Build a small SIS instance for embedding skeleton tests."""

    return Instance(
        name="small",
        index=0,
        n=2,
        m=2,
        q=7,
        gamma=7,
        A=np.array([[1, 2], [3, 4]], dtype=np.int64),
        t=t.astype(np.int64, copy=True),
        require_l2_ge_q=False,
        homogeneous=homogeneous,
        source_path=Path("memory"),
    )


def test_homogeneous_path_still_uses_existing_candidate_collection() -> None:
    """The existing homogeneous row decoder remains unchanged."""

    inst = _inst(homogeneous=True, t=np.zeros(2, dtype=np.int64))
    rows = np.array([[-1, -3, 1, 0], [-2, -4, 0, 1]], dtype=np.int64)

    cands = collect_homogeneous_candidates_from_row_basis(inst, rows, top_k=2)

    assert select_solver_path(inst) == "homogeneous"
    assert len(cands) == 2
    assert all(cand.congruence_ok for cand in cands)


def test_homogeneous_collector_still_rejects_inhomogeneous_instances() -> None:
    """The old homogeneous API still has a clear boundary."""

    inst = _inst(homogeneous=False, t=np.array([1, 2], dtype=np.int64))
    with pytest.raises(ValueError, match="inhomogeneous"):
        collect_homogeneous_candidates_from_row_basis(inst, np.eye(4, dtype=np.int64), top_k=1)


def test_kannan_embedding_basis_blocks() -> None:
    """The embedding basis has the documented block layout."""

    inst = _inst(homogeneous=False, t=np.array([1, 2], dtype=np.int64))
    B = build_kannan_embedding_basis_matrix(inst, embedding_scale=9)

    assert B.shape == (5, 5)
    np.testing.assert_array_equal(B[:2, :2], 7 * np.eye(2, dtype=np.int64))
    np.testing.assert_array_equal(B[:2, 2:4], inst.A)
    np.testing.assert_array_equal(B[:2, -1], inst.t)
    np.testing.assert_array_equal(B[2:4, 2:4], -np.eye(2, dtype=np.int64))
    assert B[-1, -1] == 9

    emb = build_kannan_embedding_row_basis(inst, embedding_scale=9)
    np.testing.assert_array_equal(emb.basis_row, B.T)
    assert emb.embedding_scale == 9


def test_embedding_decode_and_validate_reuses_existing_validator() -> None:
    """An embedding vector can be decoded and validated as an inhomogeneous candidate."""

    inst = _inst(homogeneous=False, t=np.array([1, 2], dtype=np.int64))
    v = np.array([1, 0], dtype=np.int64)
    u = inst.t - inst.A @ v
    x = np.concatenate([u, v, np.array([inst.q], dtype=np.int64)])

    decoded_u, decoded_v = decode_embedding_vector_to_uv(x, inst)
    cand = validate_embedding_vector_candidate(inst, x)

    np.testing.assert_array_equal(decoded_u, u)
    np.testing.assert_array_equal(decoded_v, v)
    assert cand.congruence_ok is True
    assert cand.valid_main is True


def test_inhomogeneous_dispatch_reaches_embedding_placeholder() -> None:
    """A nonzero target takes the embedding branch and returns the skeleton result."""

    inst = _inst(homogeneous=False, t=np.array([1, 2], dtype=np.int64))

    direct = solve_inhomogeneous_embedding_skeleton(inst)
    via_dispatch = solve_instance_baseline(inst, beta=2)

    assert select_solver_path(inst) == "inhomogeneous_embedding"
    assert direct.status == "embedding_skeleton_only"
    assert direct.candidates == []
    assert direct.embedding.basis_row.shape == (5, 5)
    assert via_dispatch.status == "embedding_skeleton_only"
