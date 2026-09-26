# Square linear solves

Include `<uni20/linalg/ops/linear_solve.hpp>`. All interfaces solve
`A X = B`, with square rank-two `A` and rank-two `B` containing one or more
right-hand sides. The scalar types must match.

## Choosing the failure policy

```cpp
auto x = uni20::linalg::solve(a, b);             // preserves both inputs
uni20::linalg::solve_inplace(a_work, b_work);    // destructive, strict

using namespace uni20::linalg;
auto info = solve_inplace_with_info(a_work, b_work);
if (!info.succeeded()) {
    // Reject this Newton step or try again with newly initialized workspaces.
}
```

`solve` and `solve_inplace` treat numerical failure as a terminal error through
Uni20's ordinary error policy. `solve_inplace_with_info` returns a `SolveInfo`
instead. All three accept an explicit backend selector as the first argument;
otherwise storage policy selects the backends. Existing async solve APIs retain
the strict policy; an async diagnostic-returning interface is not added here.

Both destructive workspaces may change on numerical failure. Only a successful
result makes `B` a valid solution. `A` contains backend-dependent elimination
data, not a portable reusable factorization. Workspaces must not overlap.
Preserving `solve` materializes column-major host matrices before dispatch.

`SolveInfo::status` is one of:

| Status | Meaning |
| --- | --- |
| `success` | A finite solution was produced (or the problem is empty). |
| `singular` | An exactly zero elimination pivot was found. |
| `small_pivot` | A nonzero pivot failed the requested relative threshold. |
| `nonfinite_input` | An input coefficient or RHS component is NaN or infinite; workspaces are preserved. |
| `nonfinite_result` | Arithmetic produced a nonfinite factor, solution, or required magnitude. |

`pivot` is an optional **zero-based elimination column**, populated only for
`singular` and `small_pivot`. It is not an original row index. Numerical
classification can differ between backends when several failure conditions
coexist: the CPU can stop during elimination, whereas LAPACK completes its LU
factorization before inspection. Pivot choices and rounding are also
backend-dependent; the tolerance applies to the pivots actually produced, not
to a backend-independent rank criterion. No residual bound or conditioning
guarantee is implied by `success`.

Zero-order systems and `N x 0` RHS matrices are successful no-ops: values are
not inspected, even if `A` is singular or nonfinite. Shape mismatch, invalid
options, unavailable backends, and invalid provider arguments remain errors,
not numerical statuses.

## Pivot policy and precision

The recoverable frontend accepts a final `SolveOptions<Real>` argument, where
`Real` is the coefficient scalar's real type (also for complex matrices):

```cpp
using Real = long double;
uni20::linalg::SolveOptions<Real> options{
    .relative_pivot_tolerance = Real{64} * uni20::numeric_limits<Real>::epsilon()
};
auto info = uni20::linalg::solve_inplace_with_info(a_work, b_work, options);
```

The default tolerance is zero, rejecting exact zero pivots only. A positive
tolerance rejects a nonzero pivot when

```text
abs(pivot[k]) / max_ij(abs(A_original[i, j])) <= relative_pivot_tolerance
```

Complex entry magnitudes are mathematical absolute values. Tolerances must be
finite and nonnegative. The comparison uses division to avoid overflowing a
product of the tolerance and matrix scale. All arithmetic and options remain
in native precision, including long double and configured MPLAPACK binary128.
This pivot heuristic is **not** a reciprocal condition estimate and is not a
rank-revealing factorization. Choosing a tolerance is an application policy;
the strict interfaces keep the zero default.

Zero is the general-purpose default because a small pivot relative to the
largest matrix entry can reflect differently scaled variables rather than an
unusable solution. For example, `A = diag(1, 1e-20)` and the column vector
`B = (1, 1e-20)` have the directly computable solution `(1, 1)`, but `64 * epsilon<double>()`
would reject its second pivot. Applications such as Newton continuation may
prefer that conservative rejection and retry with a smaller continuation step.
They should select a positive tolerance explicitly; the `64 * epsilon<Real>()`
above is an example policy, not a universal accuracy threshold. Residual and
conditioning checks remain separate application decisions.

## Backend and dispatch contract

There is one operation tag, `linear_solve_op`, with kernel arguments
`(A_descriptor, B_descriptor, SolveInfo&, SolveOptions<Real> const&)`.
Diagnostics follow the ordinary output-by-reference kernel convention.

- LAPACK uses `unchecked::getrf`, inspects the provider result and LU pivots,
  then calls `getrs`. Standard real/complex LAPACK types and configured
  MPLAPACK types use this path for compatible column-major workspaces.
- CPU reference uses accessor-respecting partial-pivoted Gaussian elimination,
  including row-major workspaces and native long double.
- A dispatch decline preserves `A`, `B`, and `SolveInfo`.
- An executed numerical failure writes `SolveInfo` and returns
  `KernelAttempt::success`. Dispatch must **not** retry another backend on
  partially overwritten workspaces.

Least squares, rank-revealing solves, condition estimates, and iterative
refinement are separate future work, not options hidden in this square solve.
