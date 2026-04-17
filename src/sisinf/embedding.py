"""Kannan embedding scaffolding for inhomogeneous SIS instances.

This module deliberately stops short of implementing a full inhomogeneous
solver. It provides the stable places where the embedding lattice is built,
where a reduced embedding vector will later be decoded back to ``(u, v)``, and
where existing validation is reused.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from sisinf.lattice import extract_row_vectors
from sisinf.types import Candidate, Instance
from sisinf.validate import validate_candidate


@dataclass(frozen=True)
class EmbeddingInstance:
    """A minimal Kannan embedding row-basis plus construction metadata."""

    basis_row: np.ndarray
    embedding_scale: int


@dataclass(frozen=True)
class EmbeddingSkeletonResult:
    """Placeholder result returned by the inhomogeneous embedding skeleton."""

    embedding: EmbeddingInstance
    candidates: list[Candidate]
    status: str


def instance_has_nonzero_target(inst: Instance) -> bool:
    """Return whether ``inst`` has an inhomogeneous nonzero target vector."""

    if inst.t is None:
        return False
    return bool(np.any(np.asarray(inst.t, dtype=np.int64).reshape(-1) != 0))


def build_kannan_embedding_basis_matrix(inst: Instance, embedding_scale: int | None = None) -> np.ndarray:
    """Build the column-basis matrix for a minimal Kannan embedding skeleton.

    The basis is ``[[qI, A, t], [0, -I, 0], [0, 0, M]]``. A coefficient vector
    ``[w; -v; 1]`` maps to ``[q w - A v + t; v; M]``. The top block is decoded
    as a candidate ``u`` and validated by the existing inhomogeneous validator:
    ``A v + u - t == 0 mod q``.

    TODO: The sign convention and embedding scale ``M`` should be revisited
    when BKZ/enumeration is wired in and empirical target-vector behavior is
    inspected.
    """

    if inst.homogeneous or not instance_has_nonzero_target(inst):
        raise ValueError(f"{inst.name} does not require Kannan embedding; target is homogeneous or zero")
    if inst.t is None:
        raise ValueError(f"{inst.name} has t=None; inhomogeneous embedding requires a target vector")

    A = np.asarray(inst.A, dtype=np.int64)
    t = np.asarray(inst.t, dtype=np.int64).reshape(-1)
    if A.shape != (inst.n, inst.m):
        raise ValueError(f"Instance A has shape {A.shape}; expected ({inst.n}, {inst.m}) for {inst.name}")
    if t.shape != (inst.n,):
        raise ValueError(f"Instance t has shape {t.shape}; expected ({inst.n},) for {inst.name}")

    scale = int(inst.q if embedding_scale is None else embedding_scale)
    if scale <= 0:
        raise ValueError(f"embedding_scale must be positive, got {scale}")

    dim = inst.n + inst.m + 1
    B = np.zeros((dim, dim), dtype=np.int64)
    B[: inst.n, : inst.n] = inst.q * np.eye(inst.n, dtype=np.int64)
    B[: inst.n, inst.n : inst.n + inst.m] = A
    B[: inst.n, -1] = t
    B[inst.n : inst.n + inst.m, inst.n : inst.n + inst.m] = -np.eye(inst.m, dtype=np.int64)
    B[-1, -1] = scale
    return B


def build_kannan_embedding_row_basis(inst: Instance, embedding_scale: int | None = None) -> EmbeddingInstance:
    """Build the fpylll-friendly row-basis for inhomogeneous Kannan embedding."""

    return EmbeddingInstance(
        basis_row=build_kannan_embedding_basis_matrix(inst, embedding_scale=embedding_scale).T.copy(),
        embedding_scale=int(inst.q if embedding_scale is None else embedding_scale),
    )


def decode_embedding_vector_to_uv(x: np.ndarray, inst: Instance) -> tuple[np.ndarray, np.ndarray]:
    """Decode an embedding vector candidate back to original ``(u, v)`` blocks.

    TODO: A real solver must decide which reduced vectors correspond to the
    embedded target layer, probably by inspecting the final coordinate near
    ``+/- embedding_scale`` and normalizing signs before this function is used.
    """

    arr = np.asarray(x, dtype=np.int64).reshape(-1)
    expected = inst.n + inst.m + 1
    if arr.shape != (expected,):
        raise ValueError(f"Embedding vector has shape {arr.shape}; expected ({expected},) for {inst.name}")
    u = arr[: inst.n].copy()
    v = arr[inst.n : inst.n + inst.m].copy()
    return u, v


def validate_embedding_vector_candidate(inst: Instance, x: np.ndarray) -> Candidate:
    """Decode one embedding vector and validate it with the existing validator."""

    u, v = decode_embedding_vector_to_uv(x, inst)
    return validate_candidate(inst, u=u, v=v)


def solve_inhomogeneous_embedding_skeleton(
    inst: Instance,
    embedding_scale: int | None = None,
) -> EmbeddingSkeletonResult:
    """Construct embedding data and return a deliberate not-yet-solved result.

    TODO: Replace this placeholder with LLL/BKZ on ``embedding.basis_row``,
    candidate extraction from reduced rows, search/ranking integration, and
    final validation against the original inhomogeneous instance.
    """

    embedding = build_kannan_embedding_row_basis(inst, embedding_scale=embedding_scale)
    # Touch row extraction to keep the skeleton exercising the same matrix
    # interface the future BKZ path will consume.
    extract_row_vectors(embedding.basis_row, limit=min(1, embedding.basis_row.shape[0]))
    return EmbeddingSkeletonResult(
        embedding=embedding,
        candidates=[],
        status="embedding_skeleton_only",
    )
