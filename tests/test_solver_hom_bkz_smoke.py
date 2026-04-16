from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

pytest.importorskip("fpylll")

from sisinf.solver_hom_bkz import solve_homogeneous_bkz_baseline  # noqa: E402
from sisinf.types import Candidate, Instance  # noqa: E402


def test_solve_homogeneous_bkz_baseline_smoke() -> None:
    """Run a tiny homogeneous baseline smoke test when fpylll is available."""

    inst = Instance(
        name="tiny",
        index=0,
        n=2,
        m=2,
        q=7,
        gamma=7,
        A=np.array([[1, 2], [3, 4]], dtype=np.int64),
        t=np.zeros(2, dtype=np.int64),
        require_l2_ge_q=False,
        homogeneous=True,
        source_path=Path("memory"),
    )
    cands = solve_homogeneous_bkz_baseline(inst, beta=2, max_loops=1, top_k=4)
    assert isinstance(cands, list)
    assert len(cands) > 0
    for cand in cands:
        assert isinstance(cand, Candidate)
        assert cand.u.shape == (inst.n,)
        assert cand.v.shape == (inst.m,)
