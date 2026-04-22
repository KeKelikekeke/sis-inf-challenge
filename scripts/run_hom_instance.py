"""Run the Stage-5 homogeneous restricted-SVP solver on one instance."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from sisinf.hom_experiments import format_run_summary, load_homogeneous_instance, run_hom_instance_workflow  # noqa: E402
from sisinf.validate import format_candidate_summary  # noqa: E402
from sisinf.two_step import TwoStepReductionTarget  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser."""

    parser = argparse.ArgumentParser(description="Run Stage-5 homogeneous SIS infinity-norm solving on one instance.")
    parser.add_argument("--input", type=Path, help="Input instance file. Can be a challenge problem file or a custom file.")
    parser.add_argument("--problem", type=int, help="Challenge problem index. Use with --input to apply official metadata.")
    parser.add_argument("--kappa", type=int, required=True, help="Sieving dimension kappa.")
    parser.add_argument("--beta", type=int, help="BKZ block size. Required by the current Stage-5 entry.")
    parser.add_argument("--target-rhf", type=float, default=1.01, help="Target RHF used by Stage-5 branching and FlexibleD4F tuning.")
    parser.add_argument("--p-success", type=float, default=0.5, help="Target success probability.")
    parser.add_argument("--q", type=int, help="Override q for custom files that do not store it.")
    parser.add_argument("--gamma", type=int, help="Override gamma for custom files that do not store it.")
    parser.add_argument("--n", type=int, help="Override n for custom files when shape inference is ambiguous.")
    parser.add_argument("--m", type=int, help="Override m for custom files when shape inference is ambiguous.")
    parser.add_argument("--a-format", choices=["auto", "row", "column"], default="auto", help="Interpretation of custom A matrices.")
    parser.add_argument("--require-l2-ge-q", action="store_true", help="Enable the extra l2 >= q condition for custom files.")
    parser.add_argument("--max-loops", type=int, default=2, help="BKZ max loops.")
    parser.add_argument("--verbose", action="store_true", help="Enable info-level logging.")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the CLI."""

    parser = build_parser()
    args = parser.parse_args(argv)
    if args.beta is None:
        parser.error("--beta is currently required because Stage-5 reduction still uses explicit BKZ beta")

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
    summary = run_hom_instance_workflow(
        inst,
        kappa=args.kappa,
        reduction_target=TwoStepReductionTarget(beta=args.beta, target_rhf=args.target_rhf, max_loops=args.max_loops),
        p_success=args.p_success,
    )

    print("Instance Summary")
    print(f"name={inst.name}")
    print(f"source_path={inst.source_path}")
    print(f"n={inst.n}, m={inst.m}, q={inst.q}, gamma={inst.gamma}")
    print(f"homogeneous={inst.homogeneous}")
    print(f"require_l2_ge_q={inst.require_l2_ge_q}")
    print()
    print("Stage-5 Result")
    print(format_run_summary(summary))
    print()
    if summary.validated_restriction_pass_candidates:
        print("Restriction-Pass Validation")
        for idx, cand in enumerate(summary.validated_restriction_pass_candidates[:3], start=1):
            print(f"candidate[{idx}]")
            print(format_candidate_summary(cand))
            print()
    else:
        print("Restriction-Pass Validation")
        print("no restriction-pass vector found")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
