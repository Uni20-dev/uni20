# Least squares

Include `<uni20/linalg/ops/least_squares.hpp>` or `<uni20/linalg/linalg.hpp>`.
The full-column-rank QR interface minimizes `||A X - B||_2` independently for each RHS
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

Both backends pack logical inputs into column-major workspaces. Triangular
substitution and residual evaluation traverse those workspaces by column;
row-major input support does not require a separate numerical implementation.
Layout-specific optimizations should follow demonstrated workload needs. Residual
evaluation reuses one length-`m` scratch vector across RHS columns.

The current provider path materializes reduced Q and projects B into a separate
`n x nrhs` workspace. It does not call `gels` or form normal equations. Applying
packed Householder reflectors directly is a possible future optimization. An
explicitly requested unavailable provider fails dispatch rather than silently
using CPU arithmetic.

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
and shape/option validation still apply. Reusable QR-factor and async interfaces
remain subsequent work in the [numerical infrastructure plan](precision_numerics_plan.md).

Run `least_squares_precision_example` for an inconsistent rectangular fit and
recoverable diagonal rejection, at native and optional arbitrary precision.

## SVD minimum-norm least squares

Include `<uni20/linalg/ops/svd_least_squares.hpp>` or the linalg umbrella. This
explicitly selected method accepts tall, square, wide and rank-deficient A:

```cpp
auto result = svd_least_squares_with_info(a, b, p,
    {.return_singular_values = true});
if (result.info.succeeded()) {
    auto const& x = *result.solution;
    auto rank = result.diagnostics->rank;
    auto cutoff = result.diagnostics->relative_cutoff;
    auto const& singular_values = result.diagnostics->singular_values;
}
auto x = svd_least_squares(a, b, p); // Strict convenience.
```

The same precision, input preservation, scalar matching and host-readable view
contracts apply as for QR. Every overload accepts an optional first backend
selector. `SvdLeastSquaresOptions<Real>` has two fields:

| Field | Default and meaning |
| --- | --- |
| `relative_cutoff` | Absent means `numeric_limits<Real>::epsilon(p)` at working precision. An explicit value must be finite and nonnegative. |
| `return_singular_values` | False. When true, retain the `min(m,n)` singular values in descending order, before truncation. |

The cutoff is relative to the largest singular value. **There is no dimension
multiplier.** A positive singular value at or below `cutoff * largest` is
excluded from the solution. Zero disables this relative truncation, but the
provider's safe-minimum safeguards still apply; it does not promise inversion
of every representable nonzero singular value. The reported cutoff is the
actual working-precision relative value, not the provider's internal absolute
underflow floor.

For the retained singular directions, the method computes the minimum-norm
least-squares solution. Rank deficiency, including rank zero, is successful.
Discarding a nonzero singular value changes the problem: the returned solution
minimizes the truncated problem and need not minimize the original residual.
The caller chooses this regularization policy. QR never switches to SVD
automatically, and the existing `least_squares` names keep their QR behavior.

`SvdLeastSquaresResult<Scalar>` contains `info`, optional `solution`,
`residual_norms`, and optional `diagnostics`. Diagnostics contain `rank`,
`relative_cutoff`, and the optional singular-value vector. Solution and
diagnostics are present exactly on success. Residuals are recomputed against
the **converted original A and B**, including directions discarded by SVD;
they have the same arithmetic limitations as the QR residuals.

In addition to `nonfinite_input` and `nonfinite_result`, SVD can return
`LeastSquaresStatus::nonconvergence`. `info.unconverged` records the provider's
count of unconverged bidiagonal off-diagonals when available. No solution,
residuals or rank diagnostics are published on failure, and dispatch does not
retry it. Shape, options, allocation and provider-availability errors remain
ordinary errors. SVD does not use the QR diagonal-rejection statuses.

Empty dimensions are supported: zero unknowns return the norms of B; zero rows
give a zero minimum-norm solution; both give rank zero. **Zero RHS columns still
compute rank from A**, unlike the QR no-op contract. They therefore inspect A
and can report nonfinite input or SVD nonconvergence.

### SVD providers and dispatch

`LapackBackend` calls `gelss`/`Rgelss`/`Cgelss`. It packs column-major input and
padded RHS work internally; callers always supply ordinary `m x nrhs` B.
Coverage is native float32/64, configured binary128, MPFR/MPC, and binary80
subject to its complex-provider restriction. The raw complex GELSS wrappers
also cover native float32/64 and binary128. This slice has no CPU-reference SVD
least-squares kernel: disabling MPLAPACK leaves native float32/64 available,
while extended types can still use the separate CPU QR method when full rank
is appropriate. No type is silently narrowed or routed through a different
algorithm.

Direct `svd_least_squares_op` dispatch takes `(solution, A, B, residual_span,
info, diagnostics, options)` and trailing precision for runtime scalars. Writable
outputs must not overlap inputs or each other. Declines preserve all arguments;
numerical failure publishes only `info`.

`rank_aware_numerics_example` demonstrates a wide minimum-norm solution, a
precision-dependent rank cutoff, and LU conditioning. Shared numerical tests
force each configured provider at float32/64/80/128 and MPFR 128/256 bits,
including supported complex types. They compare against independent rational
solutions, verify improving accuracy and test values beyond double's exponent
range.

Provider contract: [LAPACK GELSS](https://www.netlib.org/lapack/double/dgelss.f).
