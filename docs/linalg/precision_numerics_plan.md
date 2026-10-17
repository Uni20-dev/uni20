# Precision-generic numerical infrastructure

This plan connects current Bethe requirements to Uni20's dense linear algebra
and scalar-math interfaces. It builds on the standard `BlasBackend` and
`LapackBackend` families, generic `Precision` arguments and recoverable square
solves. No scalar-specific backend selectors are introduced.

## Sequence and boundaries

| Stage | Deliverable | Status |
| --- | --- | --- |
| 1 | Full-column-rank QR least squares, recoverable status and residual norms | Initial synchronous implementation available |
| 2 | SVD-based minimum-norm least squares and numerical rank | Synchronous GELSS implementation available |
| 3 | Reciprocal condition estimates from reusable LU factors | Synchronous GECON implementation available |
| 4 | QR diagnostics, backward error and iterative refinement | Design needed |
| 5 | Extract right-half-plane complex log-Gamma and log-absolute Barnes G | Separate scalar-math work |

Stage 5 is independent of the matrix work. Tridiagonal and Schur provider
coverage remains on the MPLAPACK roadmap; the immediate least-squares consumer
takes priority. Bethe owns physical convergence, continuation acceptance,
step selection and decisions to repeat at higher precision.

## 1. Full-column-rank least squares

The initial interface is documented in [least squares](least_squares.md).

Solve `min_X ||A X - B||_2` for each RHS column, initially with `m >= n`.
Inputs have one scalar type, `A` is `m x n`, `B` is `m x nrhs`, and the
solution is `n x nrhs`. Preserve both inputs and return owning host output.
Do not expose LAPACK's padded RHS storage as a Tensor API requirement.

The synchronous API is `least_squares_with_info(a, b[, p][, options])`, with
an optional first backend selector, and a strict `least_squares` convenience.
Use existing input-default precision inference and finite explicit overrides;
fixed scalars require their native precision. Exact runtime inputs need a finite
working precision. Conversion belongs to the backend, including tolerances.

Return numerical status, an optional solution and one real-valued residual norm
per RHS. Failed calculations have no published solution or residual vector.
Distinguish zero triangular diagonals, nonzero diagonals rejected by policy,
nonfinite inputs and nonfinite arithmetic. A completed numerical failure stops
dispatch: it is not a reason to retry another backend. Invalid shapes/options,
unsupported explicit selectors and allocation failures remain ordinary errors.

The optional `relative_diagonal_tolerance` defaults to zero. A positive value
rejects `abs(R[k,k]) / max(abs(A)) <= tolerance`. This is a scaling-dependent
full-rank rejection heuristic, not numerical rank or a condition estimate.
The reported column is a QR column, not a count of independent columns.
Do not import Bethe's `64 * epsilon` policy as a library default.

The first provider implementation reuses the existing reduced QR kernels,
forms `Q^H B` and solves the triangular system. It materializes Q; a subsequent
measured optimization can apply stored Householder reflectors directly. A CPU
Givens implementation supports native real/complex scalars, including long
double without MPLAPACK, and finite-precision MPFR/MPC arithmetic. Neither
implementation forms normal equations. Explicit LAPACK selection must execute
the actual configured QR provider and respect binary80 complex restrictions.

Residual norms are recomputed from the converted inputs and returned solution
at working precision using scaled norm accumulation. They are diagnostics,
not rigorous error bounds or a physical convergence criterion. Empty RHS sets
are successful no-ops; zero unknowns return the norms of B. Unsupported
underdetermined problems are rejected explicitly in this first slice.

Acceptance evidence includes rectangular inconsistent systems with exact
solutions, multiple RHS, complex conjugation, row-major and semantic views,
rank rejection, nonfinite data, zero extents and input preservation. Shared
numerical cases must cover float32/64/80/128 and MPFR at 128/256 bits, force
each supported backend, detect narrowing and verify precision-dependent
accuracy. Provider-disabled builds must retain native long-double support.

## 2. Numerical rank and minimum-norm solutions

[`svd_least_squares_with_info`](least_squares.md#svd-minimum-norm-least-squares)
selects GELSS explicitly, preserving the QR interface. It accepts rectangular,
rank-deficient and underdetermined inputs and returns a minimum-norm solution
for the retained singular directions. Rank is separate from algorithm convergence;
positive provider INFO is recoverable nonconvergence. Report the applied relative
cutoff and numerical rank, with singular values when requested.

The default relative cutoff is working-precision epsilon, without a dimension
multiplier. A finite nonnegative override is supported. Zero disables relative
truncation, while provider safe-minimum safeguards remain. Truncating a nonzero
singular value changes the least-squares problem. Zero RHS columns still compute
rank; zero rows or columns give rank zero with the documented empty solution and
residual conventions. `gelsd` and pivoted-QR `gelsy` remain optional later methods.

Native float32/64, binary128, MPFR/MPC and supported binary80 providers are
covered through `LapackBackend`. There is no CPU SVD least-squares fallback in
this slice. Provider-disabled extended types retain the CPU QR option where
full-column rank is appropriate.

## 3. LU condition estimates

[`lu_rcond_with_info`](lu.md#reciprocal-condition-estimates) estimates reciprocal
conditioning in the one- or infinity-norm through GECON, including MPFR/MPC and
supported binary80 providers. It avoids identity RHS columns and explicit
inverse construction. The result names its norm and uses factor precision.
Estimates are diagnostics, not bounds or automatic rejection criteria.

`LuOptions<Real>::retain_norms` defaults to true. LU captures both original
norms after conversion to factor precision and before factorization overwrites
A. The opt-out avoids this pass; callers can supply an original norm explicitly
later. Overflow of a diagnostic norm does not invalidate the factors. Estimation
reports an unavailable numerical result separately, preserving LU reuse.

The current condition estimator requires a configured GECON provider. Native
long-double CPU LU/solve remains available without MPLAPACK; a future CPU
condition estimator would need transposed solves and a suitable inverse-norm
algorithm. There is no silent inverse-construction or narrowing fallback.
Shared tests cover diagonal conditions, scaling, nontrivial pivoted factors,
precision-sensitive data and improved accuracy. The runnable
`rank_aware_numerics_example` covers both stages.

## 4. Further diagnostics

A triangular 1-norm/infinity-norm estimate for R is not generally the
corresponding estimate for A. For full-column-rank reduced QR, their 2-norm
condition numbers agree. State which matrix and norm each diagnostic describes.

Backward error and refinement need original A and B as well as factors. Specify
normwise versus componentwise measures and the residual-evaluation precision.
Keep evaluation precision explicit and leave automatic precision escalation
to applications. Async variants should resolve defaults after input epochs
become readable and keep every provider precision scope within one synchronous
call on its executing thread.

## 5. Complex Gamma/Barnes extraction

The model-independent `GammaBarnes<Real>` implementation in
`bethe-6spinon/include/bethe/special_functions.hpp` is the starting point.
Preserve its finite `Re(z)>0` domain and two distinct functions:

- Analytic complex log-Gamma, real on the positive axis. This is not the
  principal logarithm of the computed Gamma value.
- Real `log(abs(G(z)))`, with no implied complex-log Barnes branch.

Retain separate truncation and roundoff diagnostics, explicit evaluation
budgets, failure reporting, exact-rational Bernoulli coefficients and immutable
reusable contexts. MPFR uses internal guard bits and rounds the result to the
requested precision. Fixed types retain native arithmetic without an MPFR
dependency. Adopt current Uni20 precision queries and configuration macros.
Expose individual operations without requiring every caller to initialize
Barnes normalization merely to evaluate Gamma.

Bring the independent reference generator and precision-scaling tests, then
review the remainder estimates, cancellation behavior and output rounding.
Analytic truncation expressions evaluated in ordinary arithmetic are not
outward-rounded interval bounds. Reflection, poles and other complex domains
require a separate design. XXX transition rates, vertex formulas and model
normalizations remain in Bethe.
