# Reusable LU factors, condition estimates and logarithmic determinants

`<uni20/linalg/ops/lu.hpp>` provides synchronous, owning LU factorization and
repeated solves. `<uni20/linalg/ops/slogdet.hpp>` adds signed real or unit-phase
complex log determinants. `<uni20/linalg/ops/lu_rcond.hpp>` adds reciprocal
condition estimates. All are included by `<uni20/linalg/linalg.hpp>`.

```cpp
using namespace uni20::linalg;
auto factors = lu_factor(a);        // Preserves a; owns packed column-major L/U.
auto x = lu_solve(factors, b);       // Preserves b and factors.
auto y = lu_solve(factors, another_b);
auto determinant = slogdet(factors); // O(n), no further factorization.
// det(a) = determinant.phase * exp(determinant.log_absolute)
```

The matrix must be square. A right-hand side is a rank-two matrix, with one
column per right-hand side and the same scalar type and row count as the factors.
Inputs may be row-major or logical accessor views: factorization materializes
their values into host storage. This initial API has no asynchronous, device,
rectangular, transposed-solve or equilibration variant. Separate
[three-precision LU refinement](lu_refinement.md) composes these factors/solves
with independent working and residual arithmetic.
The runnable `lu_log_determinant_example` illustrates factor reuse and diagnostics.

## Ownership and permutation convention

`LuFactorization<Scalar>` exists only after successful factorization. It owns
its data; copies are independent. Its accessors are read-only:

- `order()` gives the matrix order.
- `packed()` returns the owning column-major matrix by const reference. The
  strict lower triangle stores L, whose unit diagonal is implicit. The upper
  triangle, including its diagonal, stores U.
- `pivots()` gives zero-based row swaps in elimination order: at step `k`,
  exchange rows `k` and `pivots()[k]`. Applying these swaps to A gives **P A = L U**.
- `norms()` returns optional original one- and infinity-norms, measured after
  conversion to factor precision; capture is enabled by default.
- `precision()` gives the factors' working precision for both fixed and runtime
  scalar types.

Borrowed matrix references and pivot spans are valid only while their factor
object retains its storage. A moved-from object supports destruction and
reassignment. Writable RHS storage must not overlap the factors.

The destructive coefficient workspace left by the older
[`solve_inplace`](linear_solve.md) interface is backend-dependent and does not
provide this reusable factor contract.

## Diagnostics and strict interfaces

`lu_factor_with_info(a)` returns `{info, factor}`; the optional factor is present
exactly on success. `lu_solve_inplace_with_info(factors, b)` overwrites an existing
RHS and returns `SolveInfo`. Strict `lu_factor`, `lu_solve` and `lu_solve_inplace`
use Uni20's terminal error policy on numerical failure. Invalid shapes, invalid
options, unavailable backends and allocation errors are ordinary errors, not
numerical statuses.

The existing `SolveStatus` vocabulary applies:

| status | meaning |
| --- | --- |
| `success` | Finite factors or solution; no conditioning guarantee. |
| `singular` | A zero pivot at the working precision. |
| `small_pivot` | A nonzero pivot rejected by the requested relative cutoff. |
| `nonfinite_input` | An input value is NaN or infinite. |
| `nonfinite_result` | Conversion or numerical arithmetic produced a nonfinite value. |

The optional `pivot` index is zero-based. Factorization takes
`LuOptions<Real>{.relative_pivot_tolerance = tolerance}`. Zero is the default,
rejecting only zero pivots. A positive tolerance rejects a pivot when its
mathematical magnitude divided by the maximum original entry magnitude is at
most the tolerance. Complex magnitudes are represented in scaled form so that
finite real and imaginary components do not overflow just while computing this
comparison. Backend pivot choices and failure precedence can differ; this is
not a rank-revealing factorization or a condition estimate.

A nonfinite RHS is rejected before modifying it. A later numerical failure may
leave a partially changed RHS. **Every RHS failure leaves the factor object
unchanged and reusable.** Nothing is retried automatically at a new precision.
A numerical failure is a completed dispatch attempt, so backend fallback does
not run on modified factorization or RHS workspaces.

## Reciprocal condition estimates

```cpp
auto factors = lu_factor(a); // Original norms retained by default.
auto result = lu_rcond_with_info(factors, MatrixNorm::One);
if (result.info.succeeded()) {
    auto reciprocal_condition = *result.estimate;
    // Application policy decides whether this sensitivity is acceptable.
}
auto rcond_inf = lu_rcond(factors, MatrixNorm::Infinity); // Strict convenience.
```

`LuConditionResult<Real>` holds `info`, the selected `norm`, and optional
`estimate`. It estimates `1 / (norm(A) * norm(inv(A)))` using GECON without
forming the inverse. The default is the induced one-norm; infinity-norm is the
other supported choice. Complex entries use mathematical magnitude, not
`abs(real) + abs(imag)`. Results use factor precision. No separate precision
argument is needed: a higher-precision estimate of lower-precision factors
would not repair those factors.

A small estimate suggests sensitivity to perturbations; it is not a forward or
backward error bound, a pivot rejection rule, or a convergence test. Zero can
be a successful estimate, including provider underflow or a condition too poor
for its safe triangular solves. Empty factors have reciprocal condition one.
The application owns any decision to recompute at higher precision.

The original norms are accumulated together in a column traversal before LU
overwrites A, after conversion to the operation precision. Capture costs O(n²)
work and O(n) temporary storage; the factor retains two real scalars. It can be
disabled when unnecessary:

```cpp
auto factors = lu_factor(a, LuOptions<Real>{.retain_norms = false});
auto estimate = lu_rcond_with_info(factors, MatrixNorm::One, original_one_norm);
```

The supplied norm must describe the original A **at factor precision**, in the
requested norm; it is converted to that precision by the backend. Calling the
retained-norm overload when capture was disabled is an ordinary error. A
nonfinite original norm or condition result reports `nonfinite_result`, with
no estimate. **Norm overflow never invalidates otherwise usable factors.**
Factorization can succeed with an infinite captured norm; solves and log
determinants remain available. A later condition-estimation failure preserves
the factor object. Negative supplied norms and unsupported norm choices are
ordinary errors.

`LapackBackend` provides GECON for native float32/64, configured binary128,
MPFR/MPC and binary80 with its existing complex-provider restriction. Any
successful Uni20 LU may supply the factors, including CPU-produced factors
when GECON supports that scalar. This slice has no CPU-reference condition
estimator. With MPLAPACK disabled, native long-double LU/solves remain available
but condition estimates for that type do not; dispatch reports unavailable
support, without narrowing or explicitly constructing an inverse as a fallback.

Direct `lu_rcond_op{norm}` dispatch takes `(estimate, packed_factors,
original_norm, SolveInfo&)`, followed by finite precision for runtime scalars.
Factors must already describe a successful LU at that precision. Declines
preserve all arguments; numerical failures modify only `SolveInfo`. Row
permutations need not be applied: both induced norms of the inverse are
invariant under the resulting column permutation. See the
[LAPACK GECON contract](https://www.netlib.org/lapack/double/dgecon.f).

Run `rank_aware_numerics_example` for conditioning, rank truncation and the norm
capture opt-out in the same precision-generic example.

## Logarithmic determinant

`slogdet(a)` factors once with norm capture disabled; `slogdet(factors)` only reduces the stored diagonal.
Both return `LogDeterminant<Scalar>{phase, log_absolute}`. The phase is a real
sign or a complex unit value, not an angle or a choice of complex logarithm
branch. Swap parity contributes its sign. The reduction sums logarithms with
compensation, normalizes complex phase products, and never forms the product of
diagonal magnitudes. A finite log determinant therefore remains useful when
the ordinary determinant would overflow or underflow.

`slogdet_with_info` additionally returns `SolveInfo` and an optional value:

| outcome | phase | log_absolute | value present? |
| --- | --- | --- | --- |
| empty 0-by-0 matrix | 1 | 0 | yes |
| nonsingular, successful | sign or unit phase | finite | yes |
| singular matrix | 0 | negative infinity | yes |
| small-pivot rejection or nonfinite failure | — | — | no |

Strict `slogdet` also returns the conventional zero determinant for singular
matrices. It uses the terminal error policy when no value can be resolved.
Singularity here is at the factorization precision; it is not a proof about an
exact matrix that was rounded before factorization.

## Precision and providers

Native scalars use their own precision. Every supported scalar also accepts an
explicit `Precision` before optional operation options; for fixed types it must
exactly match `precision_of(a)`. For `mpreal` and `complex<mpreal>`, omitting it
uses the matrix's finite construction default:

```cpp
auto p = uni20::Precision::bits(512);
auto factors = lu_factor(a, p);
auto attempt = lu_factor_with_info(a, p, LuOptions<uni20::mpreal>{});
auto determinant = slogdet(a, p);
auto x = lu_solve(factors, b); // Always uses factors.precision().
```

`factors.precision()` is available for fixed and runtime scalars alike. A generic
caller can use `lu_factor(a, precision_of(a))`; incompatible precision requests
are rejected before numerical work. Explicit precision is also supported by
the one-shot `solve`, `solve_inplace` and `solve_inplace_with_info` interfaces,
including asynchronous `solve` and `solve_inplace`.

An exact-default matrix requires an explicit finite precision. The backend
converts actual elements, regardless of their individual precisions, without
changing the source or its default. Increasing precision cannot recover digits
already lost in an approximate input.

RHS values are converted to factor precision even when their construction
default differs. Successful nonempty in-place solves record this precision on
owning RHS tensors; writing through a view leaves its owner's default unchanged.
Empty in-place solves preserve defaults. Newly returned preserving solutions
carry factor precision even when empty. Log-determinant arithmetic also uses
factor precision. There is no implicit precision escalation or exact-rational
LU in this API.

Every frontend accepts an optional first backend selector. Default preserving
operations select from the materialized host workspace:

| provider | scalars | implementation |
| --- | --- | --- |
| `CpuReferenceBackend` | native real/complex, including configured float80/float128; mpreal and complex&lt;mpreal&gt; | Accessor-respecting partial-pivot LU and triangular substitution. Runtime values convert to finite working precision without MPLAPACK. |
| `LapackBackend` | native LAPACK scalars; optional binary128 | GETRF and GETRS with direct compatible storage. |
| `LapackBackend` | mpreal and complex&lt;mpreal&gt; | Optional finite-precision MPLAPACK MPFR/MPC GETRF and GETRS, with explicit provider packing. |
| `LapackBackend` | float80; complex160 with a compatible provider type | Optional binary80 provider via value-preserving conversion; [complex arithmetic restriction](mplapack_binary80.md). |

MPFR-only builds support CPU LU, reusable solves and log determinants with the
same API. Runtime CPU kernels convert actual inputs at the selected finite
precision; they do not change a global or thread-local precision default.
Exact-default inputs still require an explicit finite operation precision.
Condition estimation remains provider-dependent. Numerical failure completes
dispatch and never retries a second backend on modified workspaces.

Providers may be mixed between factorization and reuse when their scalar
coverage permits it. Backend kernels consume the same packed factors and
zero-based pivot convention. Direct `lu_solve_op` dispatch requires already
successful factors; it does not rescan their numerical invariants on every solve.

The CPU factorization kernel also accepts direct row-major and strided workspaces.
Native kernels place the smaller-stride axis of the trailing-submatrix update
in the inner loop; mappings without stride information put rows in the inner
loop. Runtime-precision adapters pack logical values into a column-major
workspace at the operation precision, then copy results back through the
accessor. Reusable solves convert only the RHS; successful factors already
carry their working precision. Owning factors returned by `lu_factor` remain
column-major.
