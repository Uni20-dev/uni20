# src/uni20/linalg/backends/blas

This directory contains operation-tag backend adapters that route dense linalg
operations to BLAS. These adapters sit above the
[BLAS descriptor layer](../../blas/), which owns mdspan-to-BLAS
descriptor construction and direct provider calls.

The native backend adapters are GEMM and GEMV. They delegate
`try_kernel(BlasBackend, operation, ...)` to the corresponding direct
`uni20::linalg::blas::try_*` wrapper. Direct representability failures return
structured `KernelAttempt` decline reasons; provider failures remain terminal
errors. Tensor contraction is an operation-specific strategy backend above
this layer and delegates its projected rank-two operands through `gemm_op`.

When MPLAPACK MPFR is configured, `gemm.hpp` also includes `mpfr.hpp`, whose
`BlasBackend` specialization packs logical `mpreal`/`complex<mpreal>` values and
passes an explicit working precision to the provider. This specialization is
available even when native BLAS is disabled. Exact GEMM declines without touching
operands so the CPU reference kernel can perform rational arithmetic.

`binary80.hpp` supplies binary80 real/complex GEMM under the same `BlasBackend`,
including when native BLAS is disabled. Provider representation conversion stays
in the compiled MPLAPACK adapter.

## Related Documentation

- [Linalg backend source map](../)
- [BLAS/LAPACK mdspan wrappers](../../../../../docs/linalg/blas_lapack_wrappers.md)
- [Mdspan linear algebra dispatch](../../../../../docs/linalg/mdspan_dispatch.md)
