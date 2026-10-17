# Precision-generic numerical infrastructure

This plan connects current Bethe requirements to Uni20's dense linear algebra
and scalar-math interfaces. It builds on the standard `BlasBackend` and
`LapackBackend` families, generic `Precision` arguments and recoverable square
solves. No scalar-specific backend selectors are introduced.

## Sequence and boundaries

| Stage | Deliverable | Status |
| --- | --- | --- |
| 1 | Full-column-rank QR least squares, recoverable status and residual norms | Initial synchronous implementation available |
| 2 | SVD-based minimum-norm least squares and numerical rank | Planned |
| 3 | Reciprocal condition estimates from reusable LU factors | Planned |
| 4 | QR diagnostics, backward error and iterative refinement | Design needed |
| 5 | Extract right-half-plane complex log-Gamma and log-absolute Barnes G | Separate scalar-math work |

Stages 2 and 3 may be implemented together when their contracts are settled.
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

Add an explicitly selected SVD method with a documented singular-value cutoff
and minimum-norm solution semantics. Rank-deficient input can have a successful
solution; keep rank separate from algorithm convergence status. Report the
cutoff and numerical rank, with singular values where requested. Recoverable
nonconvergence needs a provider status path rather than the strict SVD wrapper.

LAPACK/MPLAPACK `gelss` is a suitable first provider; `gelsd` and pivoted-QR
`gelsy` are later alternatives, not four mandatory initial implementations.
Settle the default cutoff and underdetermined/empty semantics before coding.
See [LAPACK least squares](https://www.netlib.org/lapack/lug/node27.html).

## 3. LU condition estimates

Extend reusable LU with 1-norm/infinity-norm reciprocal condition estimates
through `gecon`, including MPFR and supported binary80 providers. This avoids
Bethe's current extra identity RHS columns and explicit inverse construction.
The original matrix norm is required: decide whether factorization retains
these norms or the estimate accepts a separately supplied norm. Name the norm
in the result and keep its arithmetic at factor precision.

CPU coverage needs either a suitable inverse-norm estimator with transposed
solves or an explicitly documented alternative. Do not silently claim provider
coverage through a fallback. Test known diagonal conditions, scaling,
nontrivial factors and precision-dependent cases. Estimates are not bounds.

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
