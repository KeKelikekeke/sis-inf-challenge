"""Scan Stage-5 homogeneous solver parameters and optionally save CSV output."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from sisinf.hom_experiments import (  # noqa: E402
    format_scan_rows,
    load_homogeneous_instance,
    scan_hom_parameter_grid,
    write_scan_csv,
)


def _parse_int_list(text: str) -> list[int]:
    return [int(part.strip()) for part in text.split(",") if part.strip()]


def _parse_float_list(text: str) -> list[float]:
    return [float(part.strip()) for part in text.split(",") if part.strip()]


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser."""

    parser = argparse.ArgumentParser(description="Scan kappa / target_rhf / p_success for Stage-5 homogeneous runs.")
    parser.add_argument("--input", type=Path, help="Input instance file. Can be a challenge problem file or a custom file.")
    parser.add_argument("--problem", type=int, help="Challenge problem index. Use with --input to apply official metadata.")
    parser.add_argument("--kappas", default="20,30", help="Comma-separated kappa values. Default: 20,30.")
    parser.add_argument("--target-rhfs", default="1.01,1.02", help="Comma-separated target RHF values. Default: 1.01,1.02.")
    parser.add_argument("--p-successes", default="0.5,0.8", help="Comma-separated target success probabilities. Default: 0.5,0.8.")
    parser.add_argument("--beta", type=int, required=True, help="BKZ block size used by the current Stage-5 entry.")
    parser.add_argument("--max-loops", type=int, default=2, help="BKZ max loops.")
    parser.add_argument("--q", type=int, help="Override q for custom files that do not store it.")
    parser.add_argument("--gamma", type=int, help="Override gamma for custom files that do not store it.")
    parser.add_argument("--n", type=int, help="Override n for custom files when shape inference is ambiguous.")
    parser.add_argument("--m", type=int, help="Override m for custom files when shape inference is ambiguous.")
    parser.add_argument("--a-format", choices=["auto", "row", "column"], default="auto", help="Interpretation of custom A matrices.")
    parser.add_argument("--require-l2-ge-q", action="store_true", help="Enable the extra l2 >= q condition for custom files.")
    parser.add_argument("--csv-out", type=Path, help="Optional CSV output path.")
    parser.add_argument("--summary-limit", type=int, default=20, help="Maximum number of rows printed to the terminal summary.")
    parser.add_argument("--verbose", action="store_true", help="Enable info-level logging.")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the CLI."""

    parser = build_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="[%(levelname)s] %(name)s: %(message)s",
    )

    inst = load_homogeneous_instance(
        input_path=args.input,
        problem_index=args.problem,
        q=args.q,
        gamma=args.gamma,
        n=args.n,
        m=args.m,
        require_l2_ge_q=args.require_l2_ge_q,
        a_format=args.a_format,
    )
    rows = scan_hom_parameter_grid(
        inst,
        kappas=_parse_int_list(args.kappas),
        target_rhfs=_parse_float_list(args.target_rhfs),
        p_successes=_parse_float_list(args.p_successes),
        beta=args.beta,
        max_loops=args.max_loops,
    )

    print("Scan Summary")
    print(f"instance_name={inst.name}")
    print(f"source_path={inst.source_path}")
    print(f"row_count={len(rows)}")
    print(format_scan_rows(rows, limit=args.summary_limit))

    if args.csv_out is not None:
        write_scan_csv(rows, args.csv_out)
        print()
        print(f"csv_out={args.csv_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
