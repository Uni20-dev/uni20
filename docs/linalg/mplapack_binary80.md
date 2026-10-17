# MPLAPACK binary80 adapter

`UNI20_ENABLE_FLOAT80=ON` together with `UNI20_BACKEND_MPLAPACK=ON`
enables real and complex binary80 GEMM
and real square solves, [reusable LU and logarithmic determinants](lu.md),
QR/LQ, [full-rank least squares](least_squares.md), self-adjoint eigensystems,
SVD and general eigensystems. Complex
LU and decompositions require a compatible provider type, as described below:

```sh
cmake -S . -B build_codex/mplapack_binary80 \
  -DUNI20_ENABLE_FLOAT80=ON -DUNI20_BACKEND_MPLAPACK=ON
```

This requires native x87 extended-precision `long double`
(`UNI20_HAS_FLOAT80=1`). It does not emulate fp80 on other platforms.
The scalar option defaults to `AUTO`; the backend defaults to `OFF`.
Binary128 and arbitrary precision are independent scalar-family choices.
The single backend switch enables kernels for all enabled families.

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

GEMM specializes `BlasBackend`; solves, LU and decompositions specialize
`LapackBackend`. Both families participate in ordinary host selection. There is
no scalar-specific backend selector. GEMM accepts logical host operands,
including conjugating accessors, and preserves BLAS zero-coefficient no-read
semantics. Configured binary80 GEMM remains available with
`UNI20_BACKEND_BLAS=OFF`, which disables the native raw BLAS wrappers.
Native CPU paths remain available where they implement the operation.

| Operation | Real provider | Complex provider |
| --- | --- | --- |
| GEMM | `Rgemm` | `Cgemm` (always) |
| LU/solve | `Rgetrf`, `Rgetrs` | `Cgetrf`, `Cgetrs` |
| Reduced QR/LQ | `Rgeqrf`/`Rorgqr`, `Rgelqf`/`Rorglq` | `Cgeqrf`/`Cungqr`, `Cgelqf`/`Cunglq` |
| Self-adjoint eigensystem | `Rsyev` | `Cheev` |
| SVD | `Rgesvd` | `Cgesvd` |
| General eigensystem | `Rgeev` | `Cgeev` |

All complex operations except GEMM are enabled only when the provider uses `long double`
(`UNI20_HAS_MPLAPACK_FLOAT80_COMPLEX=1`). With MPLAPACK 3.0.0 and GCC 13,
the distinct `_Float64x` uses generic standard-library complex division, which
squares the denominator without scaling. For example, factoring
`[[s,s],[s,2s]]` with `s=1e3000L` reports success but gives an incorrect
multiplier and a log determinant wrong by `log(2)`. Solving a one-element
system with diagonal `1e4000L` can similarly return zero for a representable
nonzero solution. This is a provider arithmetic defect, not conversion loss.

Uni20 therefore rejects complex LU/solve and decomposition provider kernels for
that provider type before any workspace is changed. Default LU/solve dispatch
selects the native CPU implementation; explicit `LapackBackend{}` selection
reports unsupported. There is no CPU decomposition fallback. Complex GEMM
remains enabled. The precision coverage report distinguishes the unsupported
provider cells from the passing CPU cases. A future provider fix requires its
own validation before this restriction can be relaxed.

The same Tensor precision interface applies to fixed and runtime scalars.
`dense_decomposition_precision_example` runs one algorithm for double, configured
binary80 and MPFR, printing reconstruction errors and spectra:

```cpp
DenseMatrix<float80> a(3, 2);
a[0, 0] = 2;
a[1, 1] = 3;
a[2, 1] = 1;
auto [q, r] = linalg::qr(a);
auto [u, s, vh] = linalg::svd(a, precision_of(a));
```

A supplied precision must equal the native 64-bit significand precision. Invalid
precision is rejected before synchronous consumption or output resizing.
Preserving, consuming, fixed-output and async decomposition interfaces use the
same contracts as the other scalars. General eigenanalysis remains synchronous
and fixed-output. SVD includes values-only, one-sided, reduced/full and
input-overwrite forms. Self-adjoint eigensystems read only the selected triangle.
The packed kernels preserve logical accessor values and support writable
row-major output storage.

The shared numerical registry exercises reconstruction, orthogonality,
eigen-residuals, retained small gaps and accuracy against a wider MPFR oracle.
Provider tests select `BlasBackend` or `LapackBackend` explicitly, so a CPU
fallback cannot certify provider coverage. Additional tests cover empty full
SVD factors, conjugate-pair unpacking, selected-triangle reads and async output.

This adapter does not make `float80` satisfy the broad raw-wrapper `BlasReal`
or `LapackReal` concepts. Kernel probes determine support per operation and
operand type. Tridiagonal eigensolvers, Schur operations, projected Krylov and
GPU coverage remain separate work.
