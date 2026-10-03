# GEMINI.md

Guidance for Gemini Code Assist reviews in this repository.

[AGENTS.md](AGENTS.md) is the canonical contributor guide. Read it and the
[review guide](docs/development/code_review.md) first. This file provides short
review-specific reminders for recurring false positives.

## Hard Project Constraints

- Uni20 is a C++23 project. Do not request C++20 compatibility changes.
- Uni20 intentionally uses multidimensional `operator[]` for tensor, matrix, and
  mdspan-style indexing. Do not request `operator()` overloads or replacements
  for compatibility with older C++ standards.
- Do not suggest defining `MDSPAN_USE_PAREN_OPERATOR`.
- Uni20 spells complex scalar types as `uni20::complex<T>` in project code,
  tests, examples, and docs. Native real types retain `std::complex<T>` identity;
  `complex<mpreal>` selects the MPC-backed owning scalar when enabled. Deduce
  generic complex arguments directly through `Complex C`; the selecting alias
  is not a template deduction surface. See the [scalar policy](docs/tensor/scalar_policy.md).
- Scalar-generic numerical code should use `uni20::numeric_limits<T>` rather
  than `std::numeric_limits<T>` directly. The primary template is undefined;
  establish the numeric category before querying limits. Runtime-precision
  quantities such as epsilon require a value or explicit precision.
- Scalar-generic host math uses `uni20::math` from `core/math.hpp`, including
  `math::abs`, `math::sqrt` and `math::abs_squared`. Its dispatch deliberately
  preserves ADL for class scalars and rejects conversion-only fallbacks.
  Classification uses `uni20::isfinite`. See the
  [scalar math contract](docs/tensor/scalar_math_design.md).

## Krylov Review Notes

- `tests/krylov/test_dense_linalg_unused.cpp` is intentionally compiled. It
  keeps quarantined dense-wrapper code tested without making those wrappers part
  of the normal Krylov API surface.
- Projected dense problems already use `uni20::DenseMatrix`; the separate
  Krylov `Matrix` type has been removed. Keep the quarantined wrapper inventory
  separate from the supported algorithms.
- Check [Krylov algorithms](docs/krylov/algorithms.md) and
  [precision validation](docs/krylov/precision_validation.md) for scalar and
  backend coverage. MPFR/MPC scalar support alone does not imply arbitrary-
  precision Krylov support.
- Vendored ARPACK references, where present in historical notes or examples,
  are comparison context. The core Uni20 target is the native matrix-free Krylov
  implementation.

## Review Priorities

Focus on correctness, numerical stability, missing tests, thread/reentrancy
issues, scalar-generic behavior, and documentation drift. Avoid compatibility
suggestions that contradict the C++23 and indexing policies above.
