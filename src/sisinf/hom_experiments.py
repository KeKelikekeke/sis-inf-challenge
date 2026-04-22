"""Engineering helpers for running and scanning homogeneous Stage-5 experiments."""

from __future__ import annotations

import csv
import json
import re
from ast import literal_eval
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from sisinf.io import (
    _extract_assignment,
    _normalize_A_from_column_format,
    _normalize_t,
    _parse_mapping,
    _strip_simple_comments,
    load_problem,
    parse_problem_file,
)
from sisinf.solver_restricted_hom import RestrictedHomogeneousSolverResult, solve_homogeneous_restricted_svp
from sisinf.two_step import ShortVectorListBackend, TwoStepReductionTarget
from sisinf.types import Candidate, Instance
from sisinf.validate import validate_candidate


@dataclass(frozen=True)
class HomInstanceRunSummary:
    """Structured Stage-5 run summary for scripts and tests."""

    instance: Instance
    reduction_target: TwoStepReductionTarget
    kappa: int
    p_success: float
    result: RestrictedHomogeneousSolverResult
    validated_restriction_pass_candidates: tuple[Candidate, ...]

    def to_record(self) -> dict[str, object]:
        """Convert the summary to a flat record for terminals or CSV."""

        return {
            "instance_name": self.instance.name,
            "source_path": str(self.instance.source_path),
            "n": self.instance.n,
            "m": self.instance.m,
            "q": self.instance.q,
            "gamma": self.instance.gamma,
            "require_l2_ge_q": self.instance.require_l2_ge_q,
            "reduction_beta": self.reduction_target.beta,
            "reduction_target_rhf": self.reduction_target.target_rhf,
            "kappa": self.kappa,
            "p_success": self.p_success,
            "selected_branch": self.result.selected_branch,
            "len_bound": self.result.len_bound,
            "single_vector_pass_probability": self.result.single_vector_pass_probability,
            "raw_required_size": self.result.raw_required_size,
            "integer_required_size": self.result.integer_required_size,
            "threshold_size": self.result.threshold_size,
            "backend_name": self.result.backend_name,
            "backend_diagnostic_only": self.result.backend_diagnostic_only,
            "produced_list_size": self.result.produced_list_size,
            "restriction_pass_count": self.result.restriction_pass_count,
            "candidate_count_after_restriction": self.result.candidate_count_after_restriction,
            "validated_pass_count": len(self.validated_restriction_pass_candidates),
        }


def _parse_custom_instance_mapping(path: Path) -> dict[str, Any]:
    """Parse a custom instance file as a mapping or assignment-style text."""

    text = _strip_simple_comments(path.read_text(encoding="utf-8"))
    mapping = _parse_mapping(text, path)
    if mapping is not None:
        if not isinstance(mapping, dict):
            raise ValueError(f"Expected mapping in {path}, got {type(mapping).__name__}")
        return dict(mapping)

    out: dict[str, Any] = {}
    for key in ("A", "t"):
        value = _extract_assignment(text, key, path)
        if value is not None:
            out[key] = value
    for key in ("n", "m", "q", "gamma", "homogeneous", "require_l2_ge_q"):
        pattern = re.compile(rf"(?m)^\s*{re.escape(key)}\s*[:=]\s*(.+?)\s*$")
        match = pattern.search(text)
        if match is not None:
            out[key] = literal_eval(match.group(1))
    if "A" not in out:
        raise ValueError(f"Missing A in custom instance file: {path}")
    return out


def _normalize_A_with_optional_shape(
    raw_A: Any,
    *,
    n: int | None,
    m: int | None,
    a_format: str,
) -> tuple[np.ndarray, int, int]:
    """Normalize a custom A matrix, inferring shape when needed."""

    arr = np.asarray(raw_A, dtype=np.int64)
    if arr.ndim != 2:
        raise ValueError(f"Custom A must be two-dimensional; got shape {arr.shape}")

    if a_format not in {"auto", "row", "column"}:
        raise ValueError(f"Unsupported a_format {a_format!r}")

    if n is not None and m is not None:
        if a_format in {"auto", "column"} and arr.shape == (m, n):
            return _normalize_A_from_column_format(arr, n=n, m=m), n, m
        if a_format in {"auto", "row"} and arr.shape == (n, m):
            return arr.copy(), n, m
        raise ValueError(f"Custom A has shape {arr.shape}; expected ({n}, {m}) for row format or ({m}, {n}) for column format")

    if a_format == "column":
        inferred_m, inferred_n = arr.shape
        return _normalize_A_from_column_format(arr, n=inferred_n, m=inferred_m), inferred_n, inferred_m

    inferred_n, inferred_m = arr.shape
    return arr.copy(), inferred_n, inferred_m


def load_homogeneous_instance(
    *,
    input_path: Path | None = None,
    problem_index: int | None = None,
    q: int | None = None,
    gamma: int | None = None,
    n: int | None = None,
    m: int | None = None,
    require_l2_ge_q: bool = False,
    a_format: str = "auto",
) -> Instance:
    """Load a homogeneous instance from a challenge file or a custom file."""

    if input_path is None and problem_index is None:
        raise ValueError("Either input_path or problem_index must be provided")

    if input_path is None:
        inst = load_problem(problem_index)
        if not inst.homogeneous:
            raise ValueError(f"problem{problem_index} is not homogeneous")
        return inst

    input_path = Path(input_path)
    if problem_index is not None:
        inst = parse_problem_file(input_path, index=problem_index)
        if not inst.homogeneous:
            raise ValueError(f"problem{problem_index} from {input_path} is not homogeneous")
        return inst

    mapping = _parse_custom_instance_mapping(input_path)
    merged_n = int(mapping.get("n", n)) if mapping.get("n", n) is not None else None
    merged_m = int(mapping.get("m", m)) if mapping.get("m", m) is not None else None
    A, merged_n, merged_m = _normalize_A_with_optional_shape(
        mapping["A"],
        n=merged_n,
        m=merged_m,
        a_format=a_format,
    )

    merged_q = mapping.get("q", q)
    merged_gamma = mapping.get("gamma", gamma)
    if merged_q is None or merged_gamma is None:
        raise ValueError("Custom instance loading requires q and gamma either in the file or via CLI")

    homogeneous = bool(mapping.get("homogeneous", True))
    if not homogeneous:
        raise ValueError("Stage-5 engineering scripts currently support homogeneous instances only")
    raw_t = mapping.get("t")
    if raw_t is None:
        t = np.zeros(merged_n, dtype=np.int64)
    else:
        t = _normalize_t(raw_t, n=merged_n, path=input_path)
        if not np.all(t == 0):
            raise ValueError("Stage-5 engineering scripts currently support homogeneous instances only; expected t = 0")

    return Instance(
        name=input_path.stem,
        index=0,
        n=merged_n,
        m=merged_m,
        q=int(merged_q),
        gamma=int(merged_gamma),
        A=A,
        t=t,
        require_l2_ge_q=bool(mapping.get("require_l2_ge_q", require_l2_ge_q)),
        homogeneous=True,
        source_path=input_path,
    )


def run_hom_instance_workflow(
    inst: Instance,
    *,
    kappa: int,
    reduction_target: TwoStepReductionTarget,
    p_success: float,
    flexible_backend: ShortVectorListBackend | None = None,
    sieve_base_backend: ShortVectorListBackend | None = None,
    sieve_upper_backend: ShortVectorListBackend | None = None,
    oversampling_constant: int = 1,
) -> HomInstanceRunSummary:
    """Run the Stage-5 homogeneous solver and validate passing vectors."""

    result = solve_homogeneous_restricted_svp(
        inst,
        reduction_target=reduction_target,
        kappa=kappa,
        p_success=p_success,
        flexible_backend=flexible_backend,
        sieve_base_backend=sieve_base_backend,
        sieve_upper_backend=sieve_upper_backend,
        oversampling_constant=oversampling_constant,
    )
    validated = []
    for vec in result.restriction_pass_vectors:
        u = np.asarray(vec, dtype=np.int64).reshape(-1)[: inst.n]
        v = np.asarray(vec, dtype=np.int64).reshape(-1)[inst.n :]
        validated.append(validate_candidate(inst, u=u, v=v))
    return HomInstanceRunSummary(
        instance=inst,
        reduction_target=reduction_target,
        kappa=kappa,
        p_success=p_success,
        result=result,
        validated_restriction_pass_candidates=tuple(validated),
    )


def scan_hom_parameter_grid(
    inst: Instance,
    *,
    kappas: list[int],
    target_rhfs: list[float],
    p_successes: list[float],
    beta: int,
    max_loops: int = 2,
    flexible_backend: ShortVectorListBackend | None = None,
    sieve_base_backend: ShortVectorListBackend | None = None,
    sieve_upper_backend: ShortVectorListBackend | None = None,
    oversampling_constant: int = 1,
) -> list[dict[str, object]]:
    """Run a small Stage-5 parameter scan and return flat records."""

    rows: list[dict[str, object]] = []
    for kappa in kappas:
        for target_rhf in target_rhfs:
            reduction_target = TwoStepReductionTarget(beta=beta, target_rhf=target_rhf, max_loops=max_loops)
            for p_success in p_successes:
                summary = run_hom_instance_workflow(
                    inst,
                    kappa=kappa,
                    reduction_target=reduction_target,
                    p_success=p_success,
                    flexible_backend=flexible_backend,
                    sieve_base_backend=sieve_base_backend,
                    sieve_upper_backend=sieve_upper_backend,
                    oversampling_constant=oversampling_constant,
                )
                rows.append(summary.to_record())
    return rows


def write_scan_csv(rows: list[dict[str, object]], output_path: Path) -> None:
    """Write scan records to CSV with a stable field order."""

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else [
        "instance_name",
        "source_path",
        "n",
        "m",
        "q",
        "gamma",
        "require_l2_ge_q",
        "reduction_beta",
        "reduction_target_rhf",
        "kappa",
        "p_success",
        "selected_branch",
        "len_bound",
        "single_vector_pass_probability",
        "raw_required_size",
        "integer_required_size",
        "threshold_size",
        "backend_name",
        "backend_diagnostic_only",
        "produced_list_size",
        "restriction_pass_count",
        "candidate_count_after_restriction",
        "validated_pass_count",
    ]
    with output_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def format_run_summary(summary: HomInstanceRunSummary) -> str:
    """Return a human-readable run summary."""

    record = summary.to_record()
    lines = [
        f"instance_name={record['instance_name']}",
        f"source_path={record['source_path']}",
        f"n={record['n']}, m={record['m']}, q={record['q']}, gamma={record['gamma']}",
        f"selected_branch={record['selected_branch']}",
        f"len_bound={record['len_bound']}",
        f"P(len)={record['single_vector_pass_probability']}",
        f"raw_required_size={record['raw_required_size']}",
        f"integer_required_size={record['integer_required_size']}",
        f"produced_list_size={record['produced_list_size']}",
        f"restriction_pass_count={record['restriction_pass_count']}",
        f"backend_name={record['backend_name']}",
        f"backend_diagnostic_only={record['backend_diagnostic_only']}",
    ]
    return "\n".join(lines)


def format_scan_rows(rows: list[dict[str, object]], *, limit: int | None = None) -> str:
    """Return a compact terminal summary for scan rows."""

    if not rows:
        return "no scan rows"
    shown = rows if limit is None else rows[:limit]
    lines = []
    for row in shown:
        lines.append(
            "kappa={kappa} target_rhf={target_rhf} p_success={p_success} branch={branch} len={len_bound} "
            "P(len)={p_single} integer_required_size={req} produced_list_size={produced} restriction_pass_count={passed} "
            "backend_diagnostic_only={diag}".format(
                kappa=row["kappa"],
                target_rhf=row["reduction_target_rhf"],
                p_success=row["p_success"],
                branch=row["selected_branch"],
                len_bound=row["len_bound"],
                p_single=row["single_vector_pass_probability"],
                req=row["integer_required_size"],
                produced=row["produced_list_size"],
                passed=row["restriction_pass_count"],
                diag=row["backend_diagnostic_only"],
            )
        )
    if limit is not None and len(rows) > limit:
        lines.append(f"... ({len(rows) - limit} more rows)")
    return "\n".join(lines)


def write_json_summary(summary: HomInstanceRunSummary, output_path: Path) -> None:
    """Write one run summary as JSON."""

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(summary.to_record())
    payload["notes"] = list(summary.result.notes)
    payload["limitations"] = list(summary.result.limitations)
    payload["validated_restriction_pass_candidates"] = [
        {
            "linf_u": cand.linf_u,
            "linf_v": cand.linf_v,
            "l2sq": cand.l2sq,
            "congruence_ok": cand.congruence_ok,
            "valid_main": cand.valid_main,
            "valid_extra": cand.valid_extra,
        }
        for cand in summary.validated_restriction_pass_candidates
    ]
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
