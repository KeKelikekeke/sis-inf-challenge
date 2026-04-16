"""Candidate validation for normalized SIS infinity-norm instances."""

from __future__ import annotations

import numpy as np

from sisinf.metrics import l2sq_int, linf_norm_int
from sisinf.types import Candidate, Instance


def validate_candidate(inst: Instance, u: np.ndarray, v: np.ndarray) -> Candidate:
    """Validate a candidate pair ``(u, v)`` against one instance."""

    u_arr = np.asarray(u, dtype=np.int64).reshape(-1)
    v_arr = np.asarray(v, dtype=np.int64).reshape(-1)

    if u_arr.shape != (inst.n,):
        raise ValueError(f"u has shape {u_arr.shape}; expected ({inst.n},) for {inst.name}")
    if v_arr.shape != (inst.m,):
        raise ValueError(f"v has shape {v_arr.shape}; expected ({inst.m},) for {inst.name}")
    if inst.A.shape != (inst.n, inst.m):
        raise ValueError(f"Instance A has shape {inst.A.shape}; expected ({inst.n}, {inst.m}) for {inst.name}")
    if inst.t is None:
        raise ValueError(f"Instance {inst.name} has t=None; loaders should normalize homogeneous t to zeros")
    t_arr = np.asarray(inst.t, dtype=np.int64).reshape(-1)
    if t_arr.shape != (inst.n,):
        raise ValueError(f"Instance t has shape {t_arr.shape}; expected ({inst.n},) for {inst.name}")

    residual = (inst.A @ v_arr + u_arr - t_arr) % inst.q
    congruence_ok = bool(np.all(residual == 0))
    linf_u = linf_norm_int(u_arr)
    linf_v = linf_norm_int(v_arr)
    l2sq = l2sq_int(u_arr) + l2sq_int(v_arr)
    valid_main = congruence_ok and linf_u <= inst.gamma and linf_v <= inst.gamma
    valid_extra = True
    if inst.require_l2_ge_q:
        valid_extra = l2sq >= inst.q * inst.q

    return Candidate(
        u=u_arr,
        v=v_arr,
        linf_u=linf_u,
        linf_v=linf_v,
        l2sq=l2sq,
        congruence_ok=congruence_ok,
        valid_main=valid_main,
        valid_extra=valid_extra,
        meta={
            "max_linf": max(linf_u, linf_v),
            "require_l2_ge_q": inst.require_l2_ge_q,
            "homogeneous": inst.homogeneous,
            "valid": valid_main and valid_extra,
        },
    )


def format_candidate_summary(cand: Candidate) -> str:
    """Format a compact human-readable validation summary."""

    valid = cand.valid_main and cand.valid_extra
    lines = [
        f"valid: {valid}",
        f"congruence_ok: {cand.congruence_ok}",
        f"linf_u: {cand.linf_u}",
        f"linf_v: {cand.linf_v}",
        f"max_linf: {cand.meta.get('max_linf')}",
        f"l2sq: {cand.l2sq}",
        f"valid_main: {cand.valid_main}",
        f"valid_extra: {cand.valid_extra}",
        f"require_l2_ge_q: {cand.meta.get('require_l2_ge_q')}",
        f"homogeneous: {cand.meta.get('homogeneous')}",
    ]
    return "\n".join(lines)
