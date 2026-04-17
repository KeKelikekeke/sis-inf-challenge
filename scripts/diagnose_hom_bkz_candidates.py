"""Run small homogeneous BKZ candidate-quality diagnostics.

The goal of this script is not benchmarking. It runs a compact grid of BKZ
parameters and compares candidate-selection mechanisms on a shared reduced
basis, so it is easier to see whether the bottleneck is BKZ strength or the
candidate pipeline.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from sisinf.io import load_problem  # noqa: E402
from sisinf.lattice import build_homogeneous_sis_row_basis, decode_lattice_vector_to_uv, extract_row_vectors  # noqa: E402
from sisinf.search import candidate_is_trivial, search_homogeneous_candidate_pool  # noqa: E402
from sisinf.solver_hom_bkz import (  # noqa: E402
    collect_homogeneous_candidates_from_row_basis,
    collect_scored_reduced_row_vectors,
    run_bkz_on_row_basis,
    run_lll_on_row_basis,
)
from sisinf.types import Candidate, Instance  # noqa: E402
from sisinf.validate import validate_candidate  # noqa: E402


@dataclass(frozen=True)
class ExperimentConfig:
    """One candidate-pipeline diagnostic configuration."""

    name: str
    use_search: bool
    selection_mode: str
    pair_max_base: int = 0
    pair_budget: int = 0
    combo_mode: str = "basic"
    combo_max_base: int = 0
    combo_budget: int = 0
    include_triples: bool = False


DEFAULT_EXPERIMENTS = [
    ExperimentConfig(name="legacy_row_prefix", use_search=False, selection_mode="legacy"),
    ExperimentConfig(name="scored_baseline", use_search=False, selection_mode="scored"),
    ExperimentConfig(
        name="scored_search_basic",
        use_search=True,
        selection_mode="scored",
        pair_max_base=20,
        pair_budget=200,
        combo_mode="basic",
    ),
    ExperimentConfig(
        name="scored_search_small_coeff",
        use_search=True,
        selection_mode="scored",
        pair_max_base=20,
        pair_budget=200,
        combo_mode="small-coeff",
        combo_max_base=4,
        combo_budget=100,
    ),
]


TABLE_COLUMNS = [
    "experiment",
    "beta",
    "max_loops",
    "top_k",
    "use_search",
    "pair_budget",
    "combo_budget",
    "candidate_count",
    "trivial_candidate_count",
    "nontrivial_candidate_count",
    "congruence_ok_count",
    "valid_extra_count",
    "valid_main_count",
    "repr_linf_u",
    "repr_linf_v",
    "repr_l2sq",
    "status",
]


def parse_int_list(text: str) -> list[int]:
    """Parse a comma-separated integer list."""

    values = [int(part.strip()) for part in text.split(",") if part.strip()]
    if not values:
        raise argparse.ArgumentTypeError("expected at least one integer")
    return values


def select_experiments(names: str | None) -> list[ExperimentConfig]:
    """Return the requested experiment configs by name."""

    if names is None:
        return DEFAULT_EXPERIMENTS
    wanted = {name.strip() for name in names.split(",") if name.strip()}
    known = {cfg.name: cfg for cfg in DEFAULT_EXPERIMENTS}
    unknown = sorted(wanted - set(known))
    if unknown:
        raise ValueError(f"Unknown experiment(s): {', '.join(unknown)}. Known: {', '.join(sorted(known))}")
    return [cfg for cfg in DEFAULT_EXPERIMENTS if cfg.name in wanted]


def validate_legacy_row_prefix(inst: Instance, B_row_red: np.ndarray, top_k: int) -> list[Candidate]:
    """Reproduce the old natural-prefix reduced-row candidate collection."""

    cands: list[Candidate] = []
    for row in extract_row_vectors(B_row_red, limit=top_k):
        u, v = decode_lattice_vector_to_uv(row, inst)
        cands.append(validate_candidate(inst, u=u, v=v))
    return cands


def representative_candidate(cands: list[Candidate]) -> Candidate | None:
    """Pick a compact representative candidate for the table."""

    for cand in cands:
        if cand.valid_main:
            return cand
    for cand in cands:
        if not candidate_is_trivial(cand):
            return cand
    return cands[0] if cands else None


def summarize_candidates(
    inst: Instance,
    cfg: ExperimentConfig,
    beta: int,
    max_loops: int,
    top_k: int,
    cands: list[Candidate],
    status: str = "ok",
) -> dict[str, Any]:
    """Build one table/CSV/JSON row."""

    del inst
    repr_cand = representative_candidate(cands)
    return {
        "experiment": cfg.name,
        "beta": beta,
        "max_loops": max_loops,
        "top_k": top_k,
        "use_search": cfg.use_search,
        "selection_mode": cfg.selection_mode,
        "pair_max_base": cfg.pair_max_base,
        "pair_budget": cfg.pair_budget,
        "combo_mode": cfg.combo_mode,
        "combo_max_base": cfg.combo_max_base,
        "combo_budget": cfg.combo_budget,
        "include_triples": cfg.include_triples,
        "candidate_count": len(cands),
        "trivial_candidate_count": sum(1 for cand in cands if candidate_is_trivial(cand)),
        "nontrivial_candidate_count": sum(1 for cand in cands if not candidate_is_trivial(cand)),
        "congruence_ok_count": sum(1 for cand in cands if cand.congruence_ok),
        "valid_extra_count": sum(1 for cand in cands if cand.valid_extra),
        "valid_main_count": sum(1 for cand in cands if cand.valid_main),
        "repr_linf_u": "" if repr_cand is None else repr_cand.linf_u,
        "repr_linf_v": "" if repr_cand is None else repr_cand.linf_v,
        "repr_l2sq": "" if repr_cand is None else repr_cand.l2sq,
        "status": status,
    }


def run_experiment_on_reduced_basis(
    inst: Instance,
    B_row_bkz: np.ndarray,
    cfg: ExperimentConfig,
    beta: int,
    max_loops: int,
    top_k: int,
) -> dict[str, Any]:
    """Run one candidate mechanism on an already reduced basis."""

    if cfg.name == "legacy_row_prefix":
        cands = validate_legacy_row_prefix(inst, B_row_bkz, top_k=top_k)
    elif cfg.use_search:
        base_vecs = collect_scored_reduced_row_vectors(
            inst,
            B_row_bkz,
            top_k=top_k,
            filter_trivial_candidates=True,
        )
        cands = search_homogeneous_candidate_pool(
            inst,
            base_vecs,
            base_top_k=top_k,
            pair_max_base=cfg.pair_max_base,
            pair_budget=cfg.pair_budget,
            filter_trivial_candidates=True,
            combo_mode=cfg.combo_mode,
            combo_max_base=cfg.combo_max_base,
            combo_budget=cfg.combo_budget,
            include_triples=cfg.include_triples,
        )
    else:
        cands = collect_homogeneous_candidates_from_row_basis(
            inst,
            B_row_bkz,
            top_k=top_k,
            filter_trivial_candidates=True,
        )
    return summarize_candidates(inst, cfg, beta, max_loops, top_k, cands)


def format_table(rows: list[dict[str, Any]], columns: list[str] = TABLE_COLUMNS) -> str:
    """Format rows as a terminal-readable table."""

    if not rows:
        return "(no rows)"
    widths = {
        col: max(len(col), *(len(str(row.get(col, ""))) for row in rows))
        for col in columns
    }
    header = " | ".join(col.ljust(widths[col]) for col in columns)
    divider = "-+-".join("-" * widths[col] for col in columns)
    body = [" | ".join(str(row.get(col, "")).ljust(widths[col]) for col in columns) for row in rows]
    return "\n".join([header, divider, *body])


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write diagnostic rows to CSV."""

    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write diagnostic rows to JSON."""

    path.write_text(json.dumps(rows, indent=2, sort_keys=True), encoding="utf-8")


def run_diagnostics(
    inst: Instance,
    betas: list[int],
    max_loops_values: list[int],
    top_k_values: list[int],
    experiments: list[ExperimentConfig],
) -> list[dict[str, Any]]:
    """Run the diagnostic grid and return summary rows."""

    rows: list[dict[str, Any]] = []
    B_row = build_homogeneous_sis_row_basis(inst)
    B_row_lll = run_lll_on_row_basis(B_row)
    for beta in betas:
        for max_loops in max_loops_values:
            B_row_bkz = run_bkz_on_row_basis(B_row_lll, beta=beta, max_loops=max_loops)
            for top_k in top_k_values:
                for cfg in experiments:
                    rows.append(run_experiment_on_reduced_basis(inst, B_row_bkz, cfg, beta, max_loops, top_k))
    return rows


def main(argv: list[str] | None = None) -> int:
    """Run the CLI."""

    parser = argparse.ArgumentParser(description="Diagnose homogeneous BKZ candidate quality across small parameter grids.")
    parser.add_argument("--problem", type=int, default=1, help="Homogeneous problem index. Default: 1.")
    parser.add_argument("--betas", type=parse_int_list, default=[10], help="Comma-separated BKZ block sizes. Default: 10.")
    parser.add_argument("--max-loops-list", type=parse_int_list, default=[1], help="Comma-separated BKZ max_loops values. Default: 1.")
    parser.add_argument("--top-k-list", type=parse_int_list, default=[10], help="Comma-separated top_k values. Default: 10.")
    parser.add_argument("--experiments", help="Comma-separated experiment names. Default: all built-in diagnostics.")
    parser.add_argument("--csv", type=Path, help="Optional CSV output path.")
    parser.add_argument("--json", type=Path, help="Optional JSON output path.")
    parser.add_argument("--verbose", action="store_true", help="Enable detailed INFO logs from candidate collection/search.")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING, format="[%(levelname)s] %(name)s: %(message)s")

    inst = load_problem(args.problem)
    if not inst.homogeneous:
        raise ValueError(f"problem{args.problem} is inhomogeneous; this diagnostic script is homogeneous-only")

    experiments = select_experiments(args.experiments)
    try:
        rows = run_diagnostics(
            inst,
            betas=args.betas,
            max_loops_values=args.max_loops_list,
            top_k_values=args.top_k_list,
            experiments=experiments,
        )
    except ImportError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    print(f"problem: {inst.index} ({inst.name}), n={inst.n}, m={inst.m}, q={inst.q}, gamma={inst.gamma}")
    print(format_table(rows))
    if args.csv:
        write_csv(args.csv, rows)
        print(f"wrote CSV: {args.csv}")
    if args.json:
        write_json(args.json, rows)
        print(f"wrote JSON: {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
