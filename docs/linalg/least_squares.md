# Full-column-rank least squares

Include `<uni20/linalg/ops/least_squares.hpp>` or `<uni20/linalg/linalg.hpp>`.
The synchronous interface minimizes `||A X - B||_2` independently for each RHS
column. `A` is `m x n` with `m >= n`, `B` is `m x nrhs`, and the owning host
solution is `n x nrhs`. Both inputs are preserved, including their precision
defaults. Real and complex inputs use the same API and must have matching scalar
types. Row-major and host-readable semantic views are read through their accessors.

```cpp
using namespace uni20::linalg;
auto attempt = least_squares_with_info(a, b);
if (attempt.info.succeeded()) {
    auto const& x = *attempt.solution;
    auto const& residuals = attempt.residual_norms;
    // residuals[j] is the Euclidean norm of A*x[:,j] - b[:,j].
}
auto x = least_squares(a, b); // Strict convenience: numerical failure is terminal.
```

`least_squares_with_info` returns `LeastSquaresResult<Scalar>` with `info`, an
optional `solution`, and `residual_norms`. The solution is present exactly on
success; failures return an empty residual vector. A successful least-squares
fit can have a large residual. The application decides whether it is useful.

## Precision and backend selection

```cpp
auto p = uni20::precision_of(a);
auto result = least_squares_with_info(a, b, p);
auto provider = least_squares_with_info(LapackBackend{}, a, b, p);
auto reference = least_squares_with_info(CpuReferenceBackend{}, a, b, p);
```

All overloads accept an optional first backend selector. Default dispatch uses
the ordinary host-result selector. Fixed scalars require their native precision;
runtime scalars require finite precision. Without an explicit request, input
defaults must agree when both are finite; exact defaults are neutral. All-exact
inputs require an explicit finite request. Backends convert actual inputs and
tolerances at the selected precision. Higher precision does not restore digits
already lost in approximate inputs. Results and residuals carry the working
precision. No process-global precision or error policy is changed.

| Backend | Implementation and coverage |
| --- | --- |
| `CpuReferenceBackend` | Givens QR and triangular substitution, native real/complex scalars and configured MPFR/MPC. Native long double needs no MPLAPACK. |
| `LapackBackend` | Existing LAPACK/MPLAPACK reduced QR, `Q^H B`, then scalar-generic triangular substitution. Its scalar coverage exactly follows QR, including the binary80 complex-provider restriction. |

The current provider path materializes reduced Q and packs logical matrices.
It does not call `gels` or form normal equations. Applying packed Householder
reflectors directly is a possible future optimization. An explicitly requested
unavailable provider fails dispatch rather than silently using CPU arithmetic.

## Numerical outcomes and diagonal policy

| Status | Meaning |
| --- | --- |
| `success` | Finite solution and residual diagnostics, or an empty RHS no-op. |
| `zero_diagonal` | A zero diagonal in the computed QR triangular factor. |
| `small_diagonal` | A nonzero diagonal rejected by the requested threshold. |
| `nonfinite_input` | A coefficient or RHS component is NaN or infinite. |
| `nonfinite_result` | Conversion, factorization, solution or a required diagnostic became nonfinite. |

`info.column` is populated only for diagonal rejection. It is a zero-based QR
column, not numerical rank. A numerical failure completes dispatch; it does not
trigger another backend attempt. Invalid shapes/options, unavailable backends
and allocation failures remain ordinary errors. Direct `least_squares_op`
dispatch takes fixed solution and residual outputs, read-only input descriptors,
`LeastSquaresInfo&` and options, with trailing precision for runtime scalars.
Outputs must not overlap the inputs or each other; declines preserve every
argument, and numerical failure publishes only the info object.

```cpp
LeastSquaresOptions<double> options{
    .relative_diagonal_tolerance = 64 * uni20::numeric_limits<double>::epsilon()
};
auto attempt = least_squares_with_info(a, b, p, options);
```

The tolerance must be finite and nonnegative. Zero is the default, rejecting
only zero diagonals. A positive tolerance rejects a diagonal when
`abs(R[k,k]) / max_ij(abs(A[i,j])) <= tolerance`, using the converted original
matrix at working precision. This is a scaling-dependent heuristic, not a
rank-revealing factorization or a condition estimate. The example threshold is
an application choice. Computed nonzero diagonals do not prove full numerical
rank, even when the exact input matrix is singular.

Residuals are recomputed from the converted original inputs and returned
solution at working precision. Norm accumulation uses `hypot` to avoid squaring
large values. Residual evaluation can still overflow or lose digits through
cancellation; it is neither a forward-error bound nor certified backward error.

Zero RHS columns succeed without inspecting values. With zero unknowns, each
residual is the norm of its RHS column; with zero rows it is zero. Finite precision
and shape/option validation still apply. Underdetermined, rank-revealing, reusable
QR-factor and async interfaces are subsequent work in the
[numerical infrastructure plan](precision_numerics_plan.md).

Run `least_squares_precision_example` for an inconsistent rectangular fit and
recoverable diagonal rejection, at native and optional arbitrary precision.
