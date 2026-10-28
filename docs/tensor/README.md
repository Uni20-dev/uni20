# Tensor Documentation

This directory owns dense Tensor value, view, scalar, creation, reshape, and
front-end operation semantics.

## Canonical Guides

- [Mdspec](mdspec.md) defines the structural
  `(data_descriptor, mapping, accessor)` representation, tensor-level view and
  lease concepts, and the initial host/CUDA acquisition APIs.
- [Tensor Operations](operations.md) defines ownership, mutation, return-value,
  materialization, and async-support contracts.
- [Scalar Tensors, Host Scalars, and Storage Transfer](scalar_tensors_and_transfer.md)
  separates rank-zero Tensor results, host control-flow values, and future
  device migration.
- [Construction, Initialization, Generated Values, and Reshape](creation_and_reshape.md)
  defines default zero-initialization, the explicit `uninitialized` API,
  generated values, layout-aware materialization, and reshape behavior.
- [Scalar Policy](scalar_policy.md) defines Uni20 scalar spellings, concepts,
  promotion boundaries, numeric limits, and optional binary128 behavior.
- [Arbitrary-Precision Scalars](mpreal.md) defines exact and finite-precision
  real/complex values, construction, arithmetic, and tensor defaults.
- [Shared Scalar Math Dispatch](scalar_math_design.md) defines the `uni20::math`
  interface, ADL customization, native precision and component-value semantics.
- [Native Special Functions and Constants](native_special_functions.md) documents
  Beta, zeta, Ei, upper incomplete Gamma, Airy Ai, reentrant log-Gamma,
  integer-order Bessel functions, typed
  constants and provider availability.
- [MPFR Special Functions](mpreal_special_functions.md) documents domains, exact
  results, constants and stable elementary extensions.

## Numerical Algorithms and Validation

These notes record native implementations, numerical acceptance criteria,
reference comparisons, and measured limitations.

- [Exponential and Logarithmic Extensions](native_exponential_extensions.md)
- [Reciprocal Functions and Catalan's Constant](native_reciprocal_math.md)
- [Compound Powers](native_compound.md)
- [Scaled Rectangular Theta Constants](elliptic_theta.md) provides native and
  explicit-precision MPFR constants without reconstructing tiny exponentials.
- [Arithmetic-Geometric Mean](native_agm.md)
- [Typed Factorial](native_factorial.md)
- [Digamma](native_digamma.md)
- [Beta](native_beta.md)
- [Riemann Zeta](native_zeta.md)
- [Exponential Integral](native_expint.md)
- [Dilogarithm](native_dilogarithm.md)
- [Upper Incomplete Gamma](native_upper_gamma.md)
- [Complex Log-Gamma and Real-Shift Ratios](complex_log_gamma.md)
- [Airy Ai](native_airy.md)
- [Finite-Interval Quadrature](finite_quadrature.md)

## Coverage Plans

- [Scalar Math Coverage](scalar_math_coverage_plan.md) tracks elementary functions,
  exact-aware utilities, special functions and provider-version decisions.

## Background Design

- [Tensor Dispatch and View Semantics Draft](dispatch_and_view_semantics_draft.md)
  preserves the reasoning that led to the current Tensor and dispatch APIs.
  Prefer the canonical guides above for implemented behavior.

Async ownership and alias lifetime are documented in
[Async Storage](../async/storage.md). Backend execution starts at the
[Kernel Dispatch](../architecture/kernel_dispatch.md) layer.

## Source Navigation

- [Tensor values, views, and operations](../../src/uni20/tensor/)
- [Storage policies](../../src/uni20/storage/)
- [Mdspan concepts and accessors](../../src/uni20/mdspan/)
- [Scalar foundations](../../src/uni20/core/)
