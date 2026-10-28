# src/uni20/core

This directory contains the smallest Uni20 type and scalar foundations. It is
intended to be cheap to include from any module and should not depend on higher
layers such as tensor, linalg, backend, async, or symmetry.

## Contents

- `types.hpp`: project scalar aliases such as `uni20::complex<T>`.
- `scalar_concepts.hpp`, `scalar_traits.hpp`, `scalar_io.hpp`: scalar
  classification, promotion, and formatting support.
- `scalar_precision.hpp`: provider-neutral runtime selection of configured real
  scalar precisions.
- `precision.hpp`: provider-independent exact or finite working precision, in every build.
- `runtime_precision.hpp`: generic `precision_of`, `native_precision` and
  `scalar_at_precision` queries/construction for fixed and runtime scalar types.
- `numeric_limits.hpp`: Uni20 numeric-limits customization point for generic
  scalar algorithms.
- `math.hpp`: shared `uni20::math` dispatch and scalar classification/construction helpers.
- `finite_quadrature.hpp`: opt-in finite tanh-sinh integration with endpoint
  distances, explicit working precision and component error diagnostics.
- `elliptic_theta.hpp`: scaled rectangular theta constants at native or explicit
  MPFR precision; see [the numerical contract](../../../docs/tensor/elliptic_theta.md).
- `complex_log_gamma.hpp`: immutable right-half-plane log-Gamma and stable
  real-shift log-ratio contexts with explicit precision and diagnostics.
- `buildinfo.hpp.in`: generated build/environment metadata.
- `dummy.*`: minimal target source used to keep build targets well-formed where
  needed.

## Notes

- Spell complex scalar types as `uni20::complex<T>` in Uni20 code.
- Use `uni20::numeric_limits<T>` in scalar-generic algorithms so non-standard
  scalar types have one project-level customization point.

## Related Documentation

- [Source tree map](../)
- [Scalar policy](../../../docs/tensor/scalar_policy.md)
- [Finite-interval quadrature](../../../docs/tensor/finite_quadrature.md)
- [Build information](../../../docs/development/build_information.md)
