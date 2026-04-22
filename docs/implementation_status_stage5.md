# Stage 5 Implementation Status

## Completed Modules

- Stage 1: restricted-SVP modeling for homogeneous SIS infinity norm.
- Stage 2: two-step scaffold, projected sublattice abstraction, pluggable short-vector backend interface.
- Stage 3: FlexibleD4F scaffold.
- Stage 4: Sieve-Then-Slice scaffold.
- Stage 5: Algorithm 8 dispatcher and homogeneous solver-facing orchestration.
- Engineering support after Stage 5:
  - homogeneous instance runner CLI
  - Stage-5 parameter scan CLI
  - CSV-oriented experiment records

## Current Mainline Capability Boundary

- The mainline now supports homogeneous SIS infinity norm only.
- The solver-facing path computes:
  - `len_bound`
  - `P(len)`
  - `raw_required_size`
  - `integer_required_size`
  - Algorithm 8 branch selection
  - final restriction scan on the produced candidate list
- The current flow is aligned with the Wang 2025 restricted-SVP mainline at the orchestration level.

## Current Limitations

- The short-vector backend is still scaffold-level and may be diagnostic-only.
- No real sieve backend is implemented yet.
- FlexibleD4F and Sieve-Then-Slice remain engineering scaffolds rather than strict paper-faithful sieve implementations.
- `P(len)` is still the Stage-1 heuristic approximation, not an exact probability law.
- The current production path does not implement:
  - Kannan embedding
  - inhomogeneous SIS infinity norm
  - a real Wang 2025 sieve output backend
- Pairwise and small-coefficient search remain diagnostic baselines only and are not part of the Stage-5 mainline.

## What Can Be Run Now

- Run the homogeneous Stage-5 solver on challenge-style or custom homogeneous inputs.
- Run small parameter scans over:
  - `kappa`
  - `target_rhf`
  - `p_success`
- Record for each run:
  - selected branch
  - `len_bound`
  - `P(len)`
  - required list sizes
  - produced list size
  - restriction-pass count
  - backend diagnostic-only status

## What Must Not Be Claimed Yet

- The repository must not claim a strict reproduction of Wang 2025 restricted-SVP solving.
- The repository must not claim a real sieve backend.
- The repository must not claim homogeneous-to-inhomogeneous coverage.
- The repository must not claim that embedding is the next practical milestone.

## Next Priority

- Highest priority: replace or supplement the current diagnostic backend with a real or experimentally meaningful short-vector backend.
- This should come before embedding work.
- In practical terms, the next stage should focus on improving candidate-list quality and list-size realism under the current homogeneous Stage-5 path.
- Until the backend improves, embedding should not be treated as the main next step because it would only stack a new problem transformation on top of an underpowered short-vector engine.
