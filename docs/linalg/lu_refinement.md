# Three-precision LU refinement

`<uni20/linalg/ops/lu_refine.hpp>` supplies synchronous dense LU refinement with
independent factor, working and residual scalar/precision roles. It is included
by `<uni20/linalg/linalg.hpp>`. Native scalars use native arithmetic; MPFR/MPC
values use explicit finite precision without changing ambient defaults.

The three precision roles follow the framework of Carson and Higham (2018),
`[CarsonHigham2018]` in the references below. The stopping and scaling
policies of this implementation are specified below.

```cpp
using namespace uni20;
using namespace uni20::linalg;
auto p = LuRefinementPrecision<float, double, long double>::native();
auto result = lu_refine_with_info(a, b, p);
if (result.info.succeeded()) {
    auto const& x = *result.solution; // double, including its accumulated updates
    auto const& diagnostic = result.columns[0].error; // long double
}
```

This factors a converted copy of A in `float`, solves corrections with those
factors, and accumulates the solution in `double`. Residuals and backward errors
are evaluated in `long double` against a separate copy of the original A and B.
On platforms where long double has the same precision as double, these two
roles have the same arithmetic. No higher precision is emulated implicitly.

Runtime precision uses the same interface:

```cpp
LuRefinementPrecision<mpreal> p{
    .factor = Precision::bits(64),
    .working = Precision::bits(128),
    .residual = Precision::bits(256),
};
auto result = lu_refine_with_info(LapackBackend{}, a, b, p);
```

The three template arguments default to the preceding scalar. Real and complex
problems use the same API, spelling complex families as `uni20::complex<Real>`.
All role and input scalars must have the same real/complex kind, but may otherwise
differ. Fixed types reject precision arguments other than their native precision.
Runtime roles reject exact precision; exact input values are permitted and
converted separately for each role. Individual input element precisions do not
replace the explicit policy.

`Precision` specifies a significand size, not an exponent range. The caller
selects the arithmetic by its scalar type. Precision ordering is not enforced;
inadequate factor or residual precision can prevent convergence or give an
optimistic evaluated residual. Increasing residual arithmetic precision cannot
recover information already lost from the supplied inputs.

## Factorization and corrections

An optional first backend selector controls LU factorization and triangular
solves using the existing dispatch. Residuals, conversions and solution updates
use host scalar arithmetic in their declared roles. There is no new numerical
provider or GPU/async path. MPLAPACK is needed only when the **factor scalar**
requires it: for example, native float factors can refine an MPFR iterate even
in a build without MPLAPACK. Native long-double factors can use CpuReferenceBackend.

The algorithm scales each RHS/residual column by a power of two toward the
upper factors' scale (capped at unity) before converting to factor arithmetic.
It undoes that shift in residual arithmetic before rounding the correction to
working precision. Using the factors' scale avoids inflating RHS values for
tiny, well-conditioned matrices until their triangular solves overflow.
Scaling avoids losing an entire small residual
solely because the factor scalar has a narrower exponent range. It does not
eliminate all componentwise underflow or guarantee convergence.
Provider limitations still apply: a triangular solve that forms a reciprocal
of a subnormal diagonal can overflow even when the mathematical solution is
finite. Such an outcome is reported as a correction failure, without fallback;
the CPU reference solve may succeed on the same factors.

Original A and B, logical input views, and supplied factors are preserved. To
reuse a factorization:

```cpp
auto result = lu_refine_with_info(factors, original_a, b, p);
```

The factors must describe the original A rounded directly to the factor scalar
and precision. This correspondence is a caller precondition; order and precision
are checked. The ordinary `lu_solve` contract still uses factor precision.
One-shot refinement does not route factor conversion through residual precision.

Temporary factors default to no norm capture. The final optional argument of the
one-shot overload is `LuOptions<FactorReal>` for pivot policy and norm capture.
No factor precision increase, backend retry after numerical failure, equilibration
or GMRES correction is performed automatically.

## Diagnostics and stopping

For each column, `SolveError<ResidualReal>` reports the infinity norm of
`r = b - A*x` and the following evaluated errors, using mathematical complex
magnitudes:

```text
normwise      = ||r||inf / (||A||inf * ||x||inf + ||b||inf)
componentwise = max_i |r_i| / (sum_k |A_ik| * |x_k| + |b_i|)
```

A zero-over-zero contribution is zero. These are diagnostics at residual
precision, not interval bounds, conditioning estimates or forward-error bounds.
Finite arithmetic is required: overflow of a product, accumulation, complex
magnitude or diagnostic denominator produces recoverable `nonfinite_result`.
Near underflow, rounded residuals and denominators can make the diagnostics
optimistic. The implementation does not claim a safe-minimum error-bound model.

`RefinementOptions<ResidualReal>` selects:

- `backward_error_tolerance`: finite and nonnegative; absent means the working
  scalar's epsilon converted to residual precision.
- `max_iterations`: maximum correction steps per RHS, default 10. Zero reports
  the initial solve and its diagnostics without corrections.

Each RHS stops independently when its componentwise error is at most the target.
Converged columns remain unchanged while other columns continue. An update that
cannot change any component of its working-precision iterate is stagnation.
Other nonconvergent behavior runs to the iteration limit; there is no hidden
rate-of-improvement threshold.

`LuRefinementResult<Working, Residual>` contains `info`, an optional owning
`solution`, and `columns`. Each column has its evaluated `error`, number of
`corrections`, and `converged`/`stagnated` indicators. `info.status` distinguishes
`converged`, `stagnated`, `iteration_limit`, `factorization_failed`,
`correction_failed`, `nonfinite_input` and `nonfinite_result`. Failed factor or
correction solves also retain their `SolveInfo` in `info.solve_info`.

The solution becomes available after the initial solve and residual evaluation
succeed. Every later update is published only after its finite values and
recomputed diagnostics succeed. Stagnation, iteration limits and later failures
therefore retain the last diagnosed finite iterate; `solution` being present
does **not** imply convergence. Columns always describe that published iterate.
Low-precision singular factors do not prove the original problem is singular.

The strict `lu_refine(...)` convenience returns the solution only on convergence
and otherwise uses the normal terminal error policy. With this forwarding
convenience, pass a named/typed options object, e.g.
`RefinementOptions<long double>{.max_iterations = 20}`.

Square A and matching rank-two RHS rows are required. Zero-order systems and
zero-column RHS sets succeed without inspecting numerical values or factoring
A. Shapes, scalar/precision policy and options are still validated. Invalid
arguments, allocation errors and unavailable backends are ordinary errors.

## Separate residual evaluation

`solve_diagnostics_with_info<Residual>(a, b, x, residual_precision)` evaluates an
existing solution without factorization or refinement. It returns `SolveInfo`,
an optional residual matrix and per-RHS `SolveError` values. All numerical outputs
are published together only on success. This permits an independent higher-
precision check of a solution from another algorithm.

`lu_refinement_example` prints the initial and refined solutions of a system with
exact solution `(2/5, -1/5)`, the three precisions, errors and correction counts.
Tests use analytic/reference solutions, cancellation controls, multiple RHS,
logical conjugation, failed lower-precision factorization and runtime precision.
Shared numerical probes cover all configured real/complex working scalar families
with explicitly selected CPU and LAPACK float32 factors; this does not certify
LU providers for the working scalar. Separate tests exercise MPFR and MPC factors.
An additional shared probe pairs every configured working scalar with binary64
residual arithmetic, checking the permitted cross-family precision conversions.

Mixed-precision GMRES, matrix-free iMPS operators and async execution are separate
work in the [numerical infrastructure plan](precision_numerics_plan.md).

## References

The first two papers provide the background for the implemented LU refinement.
The GMRES and least-squares entries support planned extensions.

| Key | Reference | Relevance |
| --- | --- | --- |
| `[Higham1997]` | N. J. Higham, ["Iterative refinement for linear systems and LAPACK"](https://doi.org/10.1093/imanum/17.4.495), *IMA Journal of Numerical Analysis* **17**(4), 495–509 (1997). | Analysis of fixed- and mixed-precision iterative refinement and its stability. |
| `[CarsonHigham2018]` | E. Carson and N. J. Higham, ["Accelerating the Solution of Linear Systems by Iterative Refinement in Three Precisions"](https://doi.org/10.1137/17M1140819), *SIAM Journal on Scientific Computing* **40**(2), A817–A847 (2018). | Three precision roles, LU-based corrections, and the GMRES-based extension; analysis of forward, normwise backward and componentwise backward errors. |
| `[AmestoyEtAl2024]` | P. Amestoy, A. Buttari, N. J. Higham, J.-Y. L'Excellent, T. Mary and B. Vieublé, ["Five-Precision GMRES-Based Iterative Refinement"](https://doi.org/10.1137/23M1549079), *SIAM Journal on Matrix Analysis and Applications* **45**(1), 529–552 (2024). | Separates precision for the preconditioned matrix-vector product from the remaining GMRES arithmetic, guiding the future GMRES precision policy. |
| `[CarsonHighamPranesh2020]` | E. Carson, N. J. Higham and S. Pranesh, ["Three-Precision GMRES-Based Iterative Refinement for Least Squares Problems"](https://doi.org/10.1137/20M1316822), *SIAM Journal on Scientific Computing* **42**(6), A4063–A4083 (2020). | QR-preconditioned augmented-system refinement for a future least-squares extension. |
