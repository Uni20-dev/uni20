# MPLAPACK Provider Adapters

This directory owns the binary80 and MPFR/MPC provider boundary used by the
ordinary BLAS and LAPACK backend families.

- `mpfr.hpp` and `mpfr.cpp` convert Uni20 scalars to provider arrays and expose
  MPFR/MPC products, solves, decompositions, and diagnostics.
- `binary80.hpp` and `binary80.cpp` provide the corresponding native extended
  precision adapters where the configured provider supports them.
- `precision_scope.hpp` establishes and restores provider precision for each
  synchronous call on the executing thread.
- `decompositions_impl.hpp` and `diagnostics_impl.hpp` share provider algorithms
  and workspace handling between scalar families.

Tensor output policy and semantic accessor packing belong above this boundary;
provider objects and precision defaults do not become public tensor types.

See [MPLAPACK and runtime precision](../../../../docs/linalg/mplapack_mpfr.md)
for configuration and contracts, and the
[dense provider inventory](../../../../docs/linalg/dense_blas_lapack_coverage.md)
for routine coverage. Return to the [backend index](../).
