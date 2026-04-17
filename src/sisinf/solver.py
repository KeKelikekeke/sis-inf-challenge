"""Top-level solver dispatch between homogeneous and inhomogeneous paths."""

from __future__ import annotations

from sisinf.embedding import EmbeddingSkeletonResult, instance_has_nonzero_target, solve_inhomogeneous_embedding_skeleton
from sisinf.solver_hom_bkz import solve_homogeneous_bkz_baseline, solve_homogeneous_bkz_with_search
from sisinf.types import Candidate, Instance


def select_solver_path(inst: Instance) -> str:
    """Return the intended solver path for one normalized instance."""

    if inst.homogeneous or not instance_has_nonzero_target(inst):
        return "homogeneous"
    return "inhomogeneous_embedding"


def solve_instance_baseline(
    inst: Instance,
    beta: int,
    max_loops: int = 2,
    top_k: int = 20,
    use_search: bool = False,
    pair_max_base: int = 20,
    pair_budget: int = 200,
    filter_trivial_candidates: bool = True,
    combo_mode: str = "basic",
    combo_max_base: int = 4,
    combo_budget: int = 100,
    include_triples: bool = False,
    embedding_scale: int | None = None,
) -> list[Candidate] | EmbeddingSkeletonResult:
    """Dispatch to the current homogeneous solver or inhomogeneous skeleton.

    TODO: Once the embedding solver is real, return a uniform result type or
    promote both branches to a shared solver-result dataclass.
    """

    if select_solver_path(inst) == "inhomogeneous_embedding":
        return solve_inhomogeneous_embedding_skeleton(inst, embedding_scale=embedding_scale)

    if use_search:
        return solve_homogeneous_bkz_with_search(
            inst,
            beta=beta,
            max_loops=max_loops,
            top_k=top_k,
            pair_max_base=pair_max_base,
            pair_budget=pair_budget,
            filter_trivial_candidates=filter_trivial_candidates,
            combo_mode=combo_mode,
            combo_max_base=combo_max_base,
            combo_budget=combo_budget,
            include_triples=include_triples,
        )
    return solve_homogeneous_bkz_baseline(
        inst,
        beta=beta,
        max_loops=max_loops,
        top_k=top_k,
        filter_trivial_candidates=filter_trivial_candidates,
    )
