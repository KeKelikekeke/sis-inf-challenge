# AGENTS.md

## Project goal

This repository is for solving SIS under infinity norm.

The main algorithmic direction is:
- reproduce Wang 2025 "Heuristic Algorithm for Solving Restricted SVP and its Applications"
- do NOT treat naive pairwise / small-coeff search as the main solver
- current pairwise code may remain only as a diagnostic baseline

## Required reading before coding

Read these files before making design decisions:
- docs/papers/wang2025_reading_notes.md
- docs/papers/wang2025_algorithm_spec.md

## Main implementation target

Implement the Wang 2025 restricted-SVP pipeline in stages:

1. restriction predicate + related probability P(len)
2. two-step solver
3. FlexibleD4F
4. Sieve-Then-Slice
5. Algorithm 8 dispatcher

Focus on homogeneous SIS∞ first.
Kannan embedding for inhomogeneous SIS∞ is stage 2.

## Hard constraints

- Do not silently replace the Wang 2025 algorithm with heuristic pairwise combinations.
- If a paper step cannot be implemented exactly, explicitly mark it as:
  - heuristic approximation
  - engineering simplification
- Keep changes modular and testable.
- Add tests and logging for every stage.
- Before large changes, first output an implementation plan.

## Validation

Every stage must include:
- unit tests
- a minimal smoke command
- a short note explaining what was implemented and what remains missing