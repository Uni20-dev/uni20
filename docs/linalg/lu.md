# Reusable LU factors and logarithmic determinants

`<uni20/linalg/ops/lu.hpp>` provides synchronous, owning LU factorization and
repeated solves. `<uni20/linalg/ops/slogdet.hpp>` adds signed real or unit-phase
complex log determinants. Both are included by `<uni20/linalg/linalg.hpp>`.

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
rectangular, transposed-solve, equilibration or iterative-refinement variant.
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
- Runtime-precision factors also expose `precision()`.

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
`SolveOptions<Real>{.relative_pivot_tolerance = tolerance}`. Zero is the default,
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

## Logarithmic determinant

`slogdet(a)` factors once; `slogdet(factors)` only reduces the stored diagonal.
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

Native scalars use their own precision. For `mpreal` and `complex<mpreal>`,
factorization uses the matrix's finite construction default, or a trailing
explicit `Precision` before optional solve options:

```cpp
auto p = uni20::Precision::bits(512);
auto factors = lu_factor(a, p);
auto attempt = lu_factor_with_info(a, p, SolveOptions<uni20::mpreal>{});
auto determinant = slogdet(a, p);
auto x = lu_solve(factors, b); // Always uses factors.precision().
```

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
| `CpuReferenceBackend` | native real/complex, including configured float80/float128 | Accessor-respecting partial-pivot LU and triangular substitution. |
| `LapackBackend` | native LAPACK scalars; optional binary128 | GETRF and GETRS with direct compatible storage. |
| `MplapackBinary80Backend` | float80; complex160 with a compatible provider type | Optional binary80 provider via value-preserving conversion; [complex arithmetic restriction](mplapack_binary80.md). |
| `MplapackMpfrBackend` | mpreal and complex&lt;mpreal&gt; | Optional finite-precision MPFR/MPC GETRF and GETRS. |

Providers may be mixed between factorization and reuse when their scalar
coverage permits it. Backend kernels consume the same packed factors and
zero-based pivot convention. Direct `lu_solve_op` dispatch requires already
successful factors; it does not rescan their numerical invariants on every solve.
