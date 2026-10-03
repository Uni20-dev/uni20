# Uni20 Copilot guidance

Read [AGENTS.md](../AGENTS.md) for contributor rules and platform scope. For code
reviews, follow the [review guide](../docs/development/code_review.md) and use
the [code-review skill](skills/code-review/SKILL.md). Canonical subsystem
documents define the intended contracts; check them against the implementation
and tests when they disagree.

Uni20 is a C++23 numerical and tensor-network library. Linux is the primary
platform; macOS CPU portability is in scope. Native Windows support is out of
scope unless explicitly requested; Windows users use WSL. Do not propose older
C++ compatibility changes or replace multidimensional `operator[]` indexing.

Use the [scalar policy](../docs/tensor/scalar_policy.md) for scalar spelling,
traits, precision and numeric limits. `uni20::complex<T>` preserves native
standard-complex identity, but selects an MPC-backed scalar for `mpreal` when
enabled. Scalar-generic code uses `uni20::numeric_limits<T>`; runtime-precision
quantities need an exemplar or an explicit precision.

For reviews:

- Prioritize numerical correctness, ownership, async causality, accessor
  semantics, symmetry preservation and backend dispatch.
- Establish a reachable failure with inputs permitted by the documented
  contract. Distinguish demonstrated defects from unresolved contract questions.
- Verify language and library semantics before alleging a compile failure.
  State which checks were actually run and which were unavailable.
- For a review-only request, report findings without editing project files or
  creating commits. Implement fixes when the maintainer also requests them.
