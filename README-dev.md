# Development Notes

This repository contains the engineering base for the 2026 SIS infinity-norm challenge.

## Stage 1: Data And Validation

Current base modules provide:

- data parsing from `data/raw_data/sis_inf_problems/problem{i}.txt`
- typed `Instance` and `Candidate` models
- candidate validation through `validate_candidate`
- repository-relative path management
- `scripts/inspect_instance.py` for instance summaries and manual candidate checks

The raw matrix `A` is provided by columns. The loader converts it to a NumPy matrix with shape `(n, m)`.

Run tests from the repository root:

```powershell
python -m pytest
```

Inspect one instance:

```powershell
python scripts/inspect_instance.py --problem 1
```

## Stage 2: Homogeneous BKZ Baseline

This stage adds a conservative homogeneous SIS infinity-norm baseline:

- `src/sisinf/lattice.py`
- `src/sisinf/solver_hom_bkz.py`
- `scripts/run_hom_bkz_smoke.py`

The homogeneous lattice basis uses the fixed convention:

```text
B = [[qI,  A],
     [ 0, -I]]
```

The mathematical basis is treated as a column basis, while fpylll reduces row bases. The implementation passes `B.T` to fpylll, and each reduced row is decoded directly as `[u; v]`. There is no sign flip on `v`.

Run a small smoke command:

```powershell
python scripts/run_hom_bkz_smoke.py --problem 1 --beta 10 --max-loops 1 --top-k 10
```

This baseline only validates candidates from the first few reduced basis rows. It is a pipeline smoke baseline, not an indication of final competition solving performance.

The BKZ path requires optional dependency `fpylll`. If it is missing, importing `sisinf` still works; LLL/BKZ calls raise an `ImportError` with installation guidance.

## Minimal Python Usage

```python
from sisinf.io import load_problem
from sisinf.solver_hom_bkz import solve_homogeneous_bkz_baseline, summarize_candidate_list

inst = load_problem(1)
cands = solve_homogeneous_bkz_baseline(inst, beta=10, max_loops=1, top_k=10)
print(summarize_candidate_list(cands))
```

## Stage 3: Homogeneous Candidate-Pool Search

This stage adds lightweight deterministic post-processing on top of the BKZ baseline:

- `src/sisinf/search.py`
- `solve_homogeneous_bkz_with_search`

The current strategy starts from reduced row-basis single vectors and adds bounded two-vector combinations with coefficients in `{-1, 0, 1}`. Each vector is decoded directly as `[u; v]`, validated with `validate_candidate`, deduplicated, and ranked by validity and simple norm metrics.

This is still a small post-processing layer, not a complete restricted-SVP solver, enumeration engine, sieve, or G6K pipeline.

Run the smoke script with search enabled:

```powershell
python scripts/run_hom_bkz_smoke.py --problem 1 --beta 10 --max-loops 1 --top-k 10 --use-search --pair-max-base 10 --pair-budget 100
```

Search scale controls:

- `top-k` controls how many reduced basis rows are kept as single-vector candidates.
- `pair-max-base` controls how many of those rows may participate in pairwise combinations.
- `pair-budget` hard-caps the number of generated pairwise combination vectors.

## Planned Modules

Next stages can add:

- `runner.py`
- `solver_inhom_embedding.py`
- stronger candidate generation / enumeration hooks
