# Precision-generic numerical infrastructure

This plan develops shared numerical infrastructure for Bethe, iMPS and other
Uni20 consumers. Dense solves are the first implementation, while iterative
refinement must also accommodate matrix-free tensor-network operators. It
builds on the standard `BlasBackend` and `LapackBackend` families, generic
`Precision` arguments and recoverable square solves. No scalar-specific backend
selectors are introduced.

## Sequence and boundaries

| Stage | Deliverable | Status |
| --- | --- | --- |
| 1 | Full-column-rank QR least squares, recoverable status and residual norms | Initial synchronous implementation available |
| 2 | SVD-based minimum-norm least squares and numerical rank | Synchronous GELSS implementation available |
| 3 | Reciprocal condition estimates from reusable LU factors | Synchronous GECON implementation available |
| 4a | Dense LU refinement with distinct factor, working and residual precisions; backward error | Initial synchronous implementation available |
| 4b | Mixed-precision GMRES and GMRES-based refinement, including matrix-free iMPS operators | Synchronous fixed right-preconditioned core available |
| 5 | Extract right-half-plane complex log-Gamma and log-absolute Barnes G | Separate scalar-math work |

Stage 5 is independent of the matrix work. Tridiagonal and Schur provider
coverage remains on the MPLAPACK roadmap. Applications own physical convergence,
continuation or tensor-update acceptance, step selection and decisions to repeat
at higher precision. Additional QR condition diagnostics can follow a concrete
consumer rather than block refinement.

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

## 4. Mixed-precision refinement and diagnostics

This is a solver layer above provider kernels. A GERFS wrapper alone does not
provide the intended precision control or matrix-free extension. No convergence
guarantee follows merely from selecting three precisions. The dense stage is
implemented in [three-precision LU refinement](lu_refinement.md).
Three precision roles apply to that algorithm without requiring GMRES.
Stage 4b now supplies restarted fixed right-preconditioned GMRES and its use
for matrix-free mixed-precision refinement corrections. See the
[GMRES contract](../krylov/gmres_refinement.md). Flexible preconditioning, adaptive
precision and asynchronous execution remain further work. Its matrix-free/iMPS
requirements below guide separation of responsibilities.
Residual evaluation, correction and updates remain separate internally.

### Precision roles

Full bibliographic entries and their roles in the implementation and follow-up
work are collected in the References section of [LU refinement](lu_refinement.md).

[Carson and Higham (2018)](https://doi.org/10.1137/17M1140819) distinguish
factorization precision, working precision for the solution and its updates,
and residual-evaluation precision. Their correction equation can use the LU
factors directly or a preconditioned GMRES solve. [Amestoy et al.
(2024)](https://doi.org/10.1137/23M1549079) extend GMRES refinement to five
precisions by separating the preconditioned matrix-vector operation from the
remaining GMRES arithmetic. The LU interface needs only the three roles;
the GMRES interface supplies its own additional policy without padding the LU
options with unused settings.

| Role | Initial dense implementation | Extension boundary |
| --- | --- | --- |
| Factorization and factor storage | LU at the selected factor scalar/precision | Other factorizations and approximate preconditioners |
| Solution and updates | Accumulate the iterate at the selected working scalar/precision | Output rounding specified independently when needed |
| Residual evaluation | Evaluate original `b - A*x` at the selected residual scalar/precision | Matrix-free application and contraction accumulation |
| Correction solve | Convert the residual explicitly and solve with existing LU factors | Krylov correction with its own arithmetic and tolerances |
| Preconditioner application | Initially the factor-precision triangular solve | Arithmetic independent of stored factor precision |

For fixed types, the scalar type selects the arithmetic; `Precision::bits(53)`
does not turn a `float` workspace into `double`. Runtime types select a finite
precision per workspace. A policy therefore needs both scalar families and
precision values, with named roles rather than an unlabelled list of bit counts.
Use configured native arithmetic for fixed types; do not silently route them
through MPFR. Distinguish stored data precision from arithmetic and accumulation
precision, and report unavailable backend combinations explicitly.

The current `lu_solve` requires matching factor/RHS scalar types and executes at
factor precision. Preserve that ordinary solve contract. The refinement layer
needs explicit workspace conversions for its initial LU correction adapter;
later higher-precision application of stored lower-precision factors needs a
separate operation contract. Neither promotion nor relabelling of factors
constitutes a new, more accurate factorization.

### Original problem and operator boundary

Retain the original A and B independently of the lower-precision factor copy.
Residual evaluation must use that original problem, not the rounded factors or
a matrix reconstructed from them. Higher-precision arithmetic can reduce error
when applying stored inputs, but cannot recover data lost before the solver was
called. The same distinction applies to rounded or truncated MPS tensors.

Separate residual evaluation, correction solving and vector updates through
Uni20's existing
[matrix-free vector boundary](../krylov/algorithms.md#matrix-free-boundary).
An operator must actually evaluate at its declared precision, not merely cast
a low-precision result. The GMRES refinement context binds distinct vector spaces
and explicit transfers. This does not extend the separate eigensolver scalar
coverage.

For iMPS transfer/environment equations, the tensor layer supplies the operator,
fixed-point projection or gauge constraint, and a compatible right-hand side.
For example, with a normalized transfer operator T and a fixed-point projector
P, an application may solve the projected problem using `I - T + P`. Increasing
arithmetic precision cannot remove the nullspace of the unprojected `I - T`.
Keep this mathematical problem definition separate from numerical preconditioning.
Preserve symmetry metadata and block structure throughout; do not materialize a
dense transfer matrix as an implicit solver fallback. See the
[uniform-MPS tangent-space notes](https://arxiv.org/abs/1810.07006) for context.

### Diagnostics and stopping

Specify normwise versus componentwise backward error, residual-evaluation
precision, per-RHS stopping and the meaning of each reported estimate before
coding the API. A matrix-free matvec alone does not provide `abs(A)*abs(x)` for
componentwise error or an operator norm for normwise error; these capabilities
must be supplied, estimated with an explicit label, or reported unavailable.
Do not substitute an unlabeled residual norm for backward error.

Distinguish target attainment, stagnation, iteration limits, correction-solver
failure and nonfinite arithmetic. Low-precision factorization failure need not
mean the original problem is singular. Inner Krylov convergence and outer
refinement convergence are different diagnostics. Recompute the final residual
of the returned iterate rather than relying only on an inner recurrence.
The dense implementation retains the last finite, diagnosed iterate on later
failure. Its default target is componentwise backward error at most working
epsilon, with at most ten corrections per RHS and configurable overrides.
No automatic precision escalation or retry is implied.

Keep operator-approximation, tensor-truncation and physical-convergence criteria
in the application. Refinement diagnostics describe the algebraic problem
actually supplied, not a certified error in the physical result.

### Implementation sequence and evidence

1. **Implemented:** `LuRefinementPrecision<Factor, Working, Residual>` selects
   finite role precisions, with native arguments required for fixed types.
   Reusable dense backward-error diagnostics and synchronous `lu_refine_with_info`
   support one-shot or supplied LU factors. Same-precision refinement is a
   special case. See the operation contract for publication and stopping rules.
2. **Implemented:** restarted GMRES and matrix-free GMRES refinement separate
   inner arithmetic, application callbacks, residual evaluation and working
   updates. BlockTensor vector operations preserve symmetry metadata. Explicit
   transfers support distinct native types and runtime precisions. Matrix-free
   residual diagnostics remain separate from dense componentwise backward error;
   optional original-operator 2-norm and conditioning information supplies
   backward-error and solution-error estimates at residual precision.
   Application-specific mixed-precision block/device transfers require their own
   adapters and validation.
3. Add async scheduling and heterogeneous execution through existing dispatch
   and scheduler mechanisms. Resolve precision defaults after input epochs are
   readable; provider precision scopes stay within one synchronous call on the
   executing thread. Explicit adaptive policies and least-squares refinement
   can follow independently.

Shared numerical tests must compare against independent higher-precision or
analytic solutions and verify the arithmetic roles, not just the final scalar
type. Cover real/complex fixed types and MPFR, successful improvement, an
insufficient residual precision, stagnation, low-precision factor failure and
input preservation. Use selected informative precision combinations rather than
an exhaustive Cartesian product. The projected transfer example exercises the
gauge constraint; the BlockTensor regression verifies that symmetry structure
survives both GMRES and refinement.

Least-squares refinement needs its own mathematical contract. [Carson, Higham
and Pranesh (2020)](https://doi.org/10.1137/20M1316822) study a QR-preconditioned
augmented-system approach; do not extend square-system formulas through normal
equations. A triangular one-/infinity-norm estimate for R is also not generally
that for A, although their two-norm condition numbers agree for full-column-rank
reduced QR. Name the matrix and norm in every future QR diagnostic.

## 5. Complex Gamma/Barnes extraction

The first extraction is implemented in
[complex log-Gamma and real-shift ratios](../tensor/complex_log_gamma.md): an
immutable right-half-plane Gamma context, shared exact Bernoulli/native Stirling
coefficients, and a paired log-ratio that avoids subtracting large phases.
Barnes G remains separate work; Gamma callers do not initialize Barnes data.

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
