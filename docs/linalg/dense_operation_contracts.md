# Dense Operation Contracts

The API comments in [`linalg/ops/`](../../src/uni20/linalg/ops/) define the
dimensions, mutation, and output meaning of each dense operation. This guide
explains how to use the fixed-output and destructive spectral interfaces.
For preserving and consuming value operations such as `eigh`, `qr`, and `svd`,
see the [Tensor operation inventory](../tensor/operations.md#dense-linear-algebra-support).

## Fixed Outputs and Backend Selection

`gemm`, `gemv`, `set_matrix`, and `matrix_exponential` write existing storage;
they do not resize it. In contrast, `assign_product` can resize an owning
output. The following GEMM overwrites `C` without reading its old elements:

```cpp
#include <uni20/linalg/ops/gemm.hpp>
#include <uni20/tensor/tensor.hpp>

uni20::DenseMatrix<double> A(2, 3), B(3, 4);
// Fill A and B with the input matrices.
uni20::DenseMatrix<double> C(uni20::uninitialized, 2, 4);
uni20::linalg::gemm(C, 1.0, A, B, 0.0); // C = A * B
```

GEMM/GEMV outputs must not overlap their inputs. Transposed and conjugated
views express the operation on the input; a provider must honor those view
semantics. A pointer-shaped storage handle alone is not enough to establish
BLAS compatibility.

An explicit selector is the first argument, for example
`gemm(CpuReferenceBackend{}, C, 1.0, A, B, 0.0)` within `uni20::linalg`.
Otherwise, the operands' storage policy selects the backend candidates.
Backend support for a scalar, layout, or memory domain is separate from the
mathematical contract. See [kernel dispatch](../architecture/kernel_dispatch.md)
for eligibility, runtime decline, and error handling, and
[provider coverage](dense_blas_lapack_coverage.md) for the implemented routines.

## Working Precision

`gemm`, `assign_product`, `add_product`, square solves, QR/SVD least squares, LU factorization, and
matrix-form logarithmic determinants, QR/LQ, self-adjoint and general eigensystems,
SVD and truncated SVD accept an explicit `Precision` for both
fixed and runtime scalar types. For example, the native GEMM above also accepts:

```cpp
auto p = uni20::precision_of(A);
uni20::linalg::gemm(C, 1.0, A, B, 0.0, p);
```

A fixed scalar requires its native precision. Synchronous calls reject
incompatible requests before output mutation or resizing. Without an explicit
request, runtime operations infer precision from participating tensor defaults,
with exact defaults neutral.
GEMM includes the output default when `beta` is nonzero. Differing finite
defaults require an explicit choice. These arguments do not extend provider
coverage to other operations. See the [generic precision contract](../tensor/scalar_policy.md#generic-precision),
[square solves](linear_solve.md#working-precision), [least squares](least_squares.md), and [LU precision policy](lu.md#precision-and-providers).

`lu_rcond` always uses its owning LU factors' precision and retained original
norms (or an explicitly supplied norm); it does not take a new precision.

For decompositions, the input matrix alone supplies the inferred precision;
preexisting output defaults do not participate. Approximation requires finite
precision, including for all-exact inputs and empty shapes. `qr(A, p)`,
`lq(A, p)`, `eigh(A, p[, triangle])`, `svd(A, p[, options])`,
`singular_values(A, p)` and `svd_left/right(A, p[, extent])` apply it explicitly.
Destructive forms place `p` before optional settings; `nonsymmetric_eigen`
places it after the required vector flag. Owning outputs and successful
workspaces record `p`; output views preserve parent defaults.
Async precision validation occurs inside the scheduled task. A failed
preserving call leaves its input readable; a failed consuming call propagates
the error to its input writer epoch as well as every result epoch. The
synchronous guarantee of rejection before consumption does not apply to a
consuming async call. See the [async failure contract](../tensor/operations.md#failures).
See [MPFR decompositions](mplapack_mpfr.md#dense-decompositions) and the
[binary80 adapter](mplapack_binary80.md) for provider
coverage, workspace handling and async behavior.

## Norms and Initialization

For a matrix `A`, `MatrixNorm` selects:

| Choice | Definition |
|---|---|
| `MaxAbs` | Largest `abs(A[i,j])`. |
| `One` | Largest column sum of `abs(A[i,j])`. |
| `Infinity` | Largest row sum of `abs(A[i,j])`. |
| `Frobenius` | Square root of the sum of `abs(A[i,j])²`. |

Here `abs` is mathematical complex magnitude, not the sum of the magnitudes
of the real and imaginary components. All four choices give zero if either
dimension is zero. For synchronous Tensor inputs, `matrix_norm(A, kind)`
returns a real rank-zero Tensor with the input's owning storage policy;
`matrix_norm_host(A, kind)` returns a host C++ scalar. These differ in result
residency, not norm definition.

For `Async<Tensor>` inputs, both functions return their respective results
wrapped in `Async`: a scalar Tensor for `matrix_norm`, and a real C++ scalar
for `matrix_norm_host`. The `_host` suffix describes result residency; it
does not imply blocking evaluation.

`set_matrix(A, diagonal, off_diagonal, region)` also accepts rectangular
matrices. `Upper` selects `row <= column`, `Lower` selects `row >= column`,
and `All` selects every entry. Unselected entries are unchanged. Thus an
identity initialization requires `All` (the default); setting only `Upper`
does not clear any existing lower-triangular entries.

`matrix_exponential(output, input, time)` computes `exp(time * input)`, not
entrywise exponentiation. Both matrices have the same square shape. Complex
time can require complex output even for real input. The current CPU
implementation copies its input before writing output, but that is not a
general overlap guarantee across backends or Async wrappers. Async fixed
input/output operations require distinct epoch queues.

## Destructive Eigensystems

These interfaces require preallocated outputs and do not preserve their
input workspaces. Make a copy first when the original matrix is still needed.

| Operation | Input/workspace | Successful output |
|---|---|---|
| `symmetric_tridiagonal_eigen` | Real diagonal of length `n` and subdiagonal of length `n-1` (both empty for `n=0`). | Diagonal becomes ascending eigenvalues; subdiagonal is destroyed. Requested `n x n` vectors are orthonormal columns. |
| `nonsymmetric_eigen` | Real or complex `n x n` matrix, destroyed. | Length-`n` complex eigenvalues and optional `n x n` complex right eigenvectors. |
| `schur` | Real or complex `n x n` matrix, overwritten by `T`. | Length-`n` complex eigenvalues and optional `n x n` Schur vectors `Q`. |
| `hessenberg_schur` | Real upper-Hessenberg `n x n` matrix, overwritten by `T`. | Same output interpretation as real `schur`, relative to the supplied Hessenberg matrix. |

For nonsymmetric eigenanalysis, column `j` is the right eigenvector satisfying
`A * v = eigenvalues[j] * v`, with unit Euclidean norm. The eigenvalues are not
sorted. Real input produces consecutive conjugate pairs, positive imaginary
part first. Uni20 unpacks the provider's real representation into ordinary
complex columns. See [LAPACK GEEV](https://www.netlib.org/lapack/double/dgeev.f)
for the underlying normalization and pair convention.

For tridiagonal input, vectors belong to the supplied tridiagonal matrix;
they do not include any earlier dense-to-tridiagonal transformation. Likewise,
`hessenberg_schur` initializes its own `Q` instead of accumulating a prior
Hessenberg reduction. Its input must already have zeros below the first
subdiagonal. [LAPACK STEQR](https://www.netlib.org/lapack/double/dsteqr.f)
describes the tridiagonal eigenvalue ordering and vector convention.

When the vector flag is false, vector elements are unchanged and the vector
shape is ignored. The API still requires a tensor argument of a supported
type: a `0 x 0` host tensor of the appropriate scalar type is suitable for
the LAPACK path. It is not a null-pointer overload. Workspaces and requested
outputs must not overlap.

Convergence failures use Uni20's error policy. These interfaces do not return
partial-result diagnostics or roll back overwritten workspaces. Do not treat
their output buffers as successful results after an error.

## Schur Vectors and Reordering

Schur decomposition gives `A = Q * T * adjoint(Q)`. For real input, `Q` is
orthogonal and `T` is upper quasi-triangular with real 1x1 and 2x2 diagonal
blocks. For complex input, `Q` is unitary and `T` is upper triangular.
Schur vectors are generally different from eigenvectors; no eigenvalue
sorting is requested by the decomposition.

`reorder_schur(T, Q, from, to, update_vectors)` moves a diagonal block using
**zero-based matrix row indices**, not ordinal block numbers. The form must
already be in Schur canonical form, as returned by `schur`. For real 2x2
blocks, either row selects the block, and the final start row can differ from
`to` because of block sizes. Uni20 does not return LAPACK's adjusted index.
Complex forms move individual diagonal entries. Both indices must be less
than the matrix order; an empty form has no valid reordering call.

With vector updates enabled, supply the existing Schur vectors in `Q`; they
change along with `T` to preserve the
decomposition of the original matrix. With updates disabled, its elements
are unchanged. Any separately saved eigenvalue array is also unchanged and
must not be assumed to match the new block order. A failed real block swap
may leave a partially reordered form. These are the semantics of
[LAPACK TREXC](https://www.netlib.org/lapack/double/dtrexc.f), with indices
translated from one-based provider indexing at the Uni20 boundary.
