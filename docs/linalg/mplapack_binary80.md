# MPLAPACK binary80 adapter

`UNI20_ENABLE_MPLAPACK_BINARY80=ON` enables real and complex binary80 GEMM
and real square solves, [reusable LU and logarithmic determinants](lu.md).
Complex LU/solve provider support depends on the provider type, as described below:

```sh
cmake -S . -B build_codex/mplapack_binary80 \
  -DUNI20_ENABLE_MPLAPACK_BINARY80=ON
```

This requires native x87 extended-precision `long double`
(`UNI20_HAS_FLOAT80=1`). It does not emulate fp80 on other platforms. The option
is independent of binary128 (`UNI20_ENABLE_MPLAPACK`) and MPFR/MPC
(`UNI20_ENABLE_MPLAPACK_MPFR`); any combination can be enabled.

Dependency resolution uses MPLAPACK >= 3.0.0 with the
`mplapack::mplapack_binary80` target, or the pinned v3.0.0 source. The existing
`UNI20_USE_SYSTEM_MPLAPACK=AUTO|ON|OFF` policy applies. The fetched reference
provider is built without the optimized, CUDA or OpenCL variants.

Uni20 keeps `float80 = long double` and `complex160 = complex<float80>`.
MPLAPACK may use the distinct GCC `_Float64x` type, despite its identical
numerical format. Its type stays private to the compiled adapter. Arrays are
converted by value, including complex components, without a double-precision
intermediate or pointer reinterpretation. Configure-time checks verify the
provider format. This implementation packs logical host values and then packs
provider arrays; it is not a zero-copy adapter.

`MplapackBinary80Backend{}` is included in the default host selector when enabled.
It accepts logical host GEMM operands, including conjugating accessors, and
preserves BLAS zero-coefficient no-read semantics. Native CPU paths remain
available through ordinary dispatch fallback or explicit selection.

The unconditional provider coverage is `Rgemm`, `Cgemm`, `Rgetrf` and `Rgetrs`.
`Cgetrf` and `Cgetrs` are enabled only when the provider uses `long double`
(`UNI20_HAS_MPLAPACK_BINARY80_COMPLEX_LU=1`). With MPLAPACK 3.0.0 and GCC 13,
the distinct `_Float64x` uses generic standard-library complex division, which
squares the denominator without scaling. For example, factoring
`[[s,s],[s,2s]]` with `s=1e3000L` reports success but gives an incorrect
multiplier and a log determinant wrong by `log(2)`. Solving a one-element
system with diagonal `1e4000L` can similarly return zero for a representable
nonzero solution. This is a provider arithmetic defect, not conversion loss.

Uni20 therefore declines complex LU/solve for that provider type, before any
workspace is changed. Default dispatch selects the native CPU implementation;
explicit `MplapackBinary80Backend{}` selection reports unsupported. Complex GEMM
remains enabled. The precision coverage report distinguishes the unsupported
provider cells from the passing CPU cases. A future provider fix requires its
own validation before this restriction can be relaxed.

This adapter does not make `float80` satisfy the broad `BlasReal` or
`LapackReal` concepts, and does not extend projected Krylov solvers, SVD,
eigensystems or GPU coverage. Those require their own provider wiring and tests.
