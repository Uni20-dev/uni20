---
name: code-review
description: Review Uni20 pull requests and C++ changes for numerical correctness, regressions, ownership, async causality, symmetry preservation, backend dispatch and missing evidence. Use for code review and re-review after updates.
---

# Uni20 code review

Read [AGENTS.md](../../../AGENTS.md) and apply the canonical
[review guide](../../../docs/development/code_review.md). This skill routes
Copilot reviews to the shared project policy.

## Establish the contract

Inspect the current PR head, the diff, changed tests and nearby callers. Identify
the maintainer-approved behavior before judging the implementation. Read the
relevant subsystem documents:

- Scalar aliases, complex deduction and runtime precision:
  [scalar policy](../../../docs/tensor/scalar_policy.md).
- Numerical test oracles, precision retention and backend coverage:
  [numerical validation](../../../docs/development/numerical_testing.md).
- Dense linalg ownership, layouts, diagnostics and async support:
  [operation contracts](../../../docs/linalg/dense_operation_contracts.md).
- Krylov algorithms and supported precision paths:
  [algorithms](../../../docs/krylov/algorithms.md) and
  [precision validation](../../../docs/krylov/precision_validation.md).

Use the review guide's numerical, tensor/accessor, async and symmetry checklists
where the change affects those contracts. Do not infer provider or algorithm
support merely from a scalar trait or from another backend passing its tests.

## Verify findings

For each candidate defect, trace a concrete permitted input through the relevant
call path. Check earlier validation and established invariants. Carrier extrema
alone are not evidence of reachable tensor dimensions or allocation sizes;
parsers, metadata, provider integer conversions and ordinary valid arithmetic
still need overflow scrutiny under the shared reachability policy.

Use available tools for focused reproductions or independent numerical checks.
Follow AGENTS.md and any local environment overrides when building. Preserve
the tested precision in inputs, tolerances and error measurements. Distinguish
executed evidence from source inspection and unavailable checks.

On re-review, check previous findings against the current head. Report a
persisting issue with current evidence; a resolved thread or an old review is
not itself evidence that the defect is present or fixed.

## Report

Lead with actionable findings ordered by severity. Give the affected file and
line, triggering input or precondition, observed or established failure, and
its consequence. Identify uncertain contracts as questions. Avoid padding a
review with speculative or cosmetic findings; report no actionable findings
when that is the result, with any material validation limits.

A review-only request calls for findings, not edits or commits. Carry out fixes
only when the maintainer also requests implementation.
