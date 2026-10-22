# Matrix-free GMRES and refinement

`krylov/gmres.hpp` implements synchronous restarted GMRES with a fixed right
preconditioner. `krylov/gmres_refinement.hpp` uses it for residual refinement
with independently typed working, inner and residual vector spaces. Both retain
opaque application vectors: no indexing, host pointer or dense transfer matrix
is required by the solver.

## Standalone solve

```cpp
auto info = uni20::krylov::gmres_with_info<double>(
    vector_ops, operator_ops, preconditioner, x, b,
    {.relative_tolerance = 1e-10, .restart = 30, .max_iterations = 1000});
```

Omit `preconditioner` to use the identity. The explicit-precision overload takes
a trailing `Precision`, after the options. It works for fixed types when the
precision matches their native format; runtime types require finite precision.
Real and complex float32/64/80/128 and configured MPFR/MPC are supported without
requiring a projected LAPACK eigensolver. Exact-only arithmetic is not a GMRES
working precision.

`vector_ops` implements `KrylovVectorOps`: allocation, independent copying,
zeroing, scaling, AXPY, dimension queries and a conjugate-linear-first inner
product. An optional `norm` supplies a Euclidean norm directly; otherwise the
solver derives one from the inner product. Operators expose
`matvec(output, input)` and preconditioners expose `apply(output, input)`.
Callbacks receive distinct input and output vectors and overwrite the output.
They must expose nonfinite data through their reductions, rather than hide it
as a zero norm. A robust norm implementation is recommended for a wide exponent
range; the inner-product fallback can overflow or underflow while squaring
finite entries.

The declared precision controls the solver's host scalar arithmetic. Vector
operations must implement the corresponding arithmetic themselves, and inputs
must be compatible with that vector space. The solver cannot inspect or round
elements inside an opaque vector. For example, initialize the dense MPFR
prototype's input values at the intended precision; a precision argument alone
does not change the arithmetic implemented by an application callback.

Operations choose their own backends. `BlockTensorVectorOps` preserves quantum
numbers, boundaries and the stored-block pattern. `DenseHostVectorAlgebra` is a
small host adapter without an assembled operator; `DenseHostVectorOps` adds a
dense reference operator for tests and small experiments.

The supplied `x` is the initial guess. An already-converged guess returns with
zero Arnoldi iterations. A zero iteration budget evaluates its residual and
does not change it. The RHS is preserved internally even if it aliases `x`.

### Arithmetic and memory

Each restart cycle uses modified Gram-Schmidt with a selective second pass.
Zero-component complex Givens rotations and zero Arnoldi extensions are handled
without division by zero. A zero extension is successful only when the candidate
passes the true-residual test; otherwise it is reported as breakdown.

The preconditioner must represent a fixed linear inverse operation throughout
one GMRES call. The algorithm stores the Arnoldi basis and applies the
preconditioner to the combined basis correction at the end of a cycle. It does
not retain a second basis of preconditioned vectors. Besides the basis, storage
is a fixed number of application vectors and a small Hessenberg matrix.

Variable/nonlinear preconditioners, including an inner iterative solve whose
effective map changes between applications, need flexible GMRES. That is future
work, with an explicit additional basis-storage cost. Changing arithmetic
precision adaptively within a solve is likewise not part of this interface.

### Stopping and publication

The stopping criterion is

```text
||b - A*x||_2 <= max(absolute_tolerance, relative_tolerance * ||b||_2).
```

The default relative tolerance is `sqrt(epsilon)` at the GMRES working precision;
the absolute tolerance is zero. Restart length defaults to 30, capped by problem
dimension and iteration budget. The maximum number of Arnoldi steps is 1000.
Zero RHS uses the absolute target, without dividing by zero.

The Hessenberg residual estimate triggers a candidate evaluation. Convergence
always requires recomputing the original residual. A failed estimate check
restarts from that recomputed residual. `GmresInfo::residual` contains the norm
and RHS and solution norms for the last successfully diagnosed, published
iterate. It is absent if initial diagnosis fails. The counters distinguish attempted Arnoldi
steps, started cycles, operator applications (including residual checks), and
preconditioner applications (including the combined correction).

Statuses distinguish convergence, iteration limit, stagnation (an update cannot
change the working iterate), breakdown, nonfinite input and nonfinite results.
A later numerical failure leaves the last published iterate and its diagnostic
intact. Invalid configuration throws before numerical work; application callback
exceptions propagate. These are recoverable numerical interfaces; applications
choose how to report unsuccessful results.

## Optional operator information and error estimates

The application can supply `GmresOperatorEstimates<Real>` through
`options.operator_estimates`. All quantities use the **2-norm induced by the
vector-space inner product**. They describe the original operator actually
solved, including a fixed-point projector or gauge constraint. They must not
describe just the preconditioner, a differently scaled problem, or the small
Hessenberg matrix. LU's separate 1-/infinity-norm diagnostics are unchanged.

```cpp
using Estimate = uni20::krylov::GmresOperatorEstimate<double>;
auto info = uni20::krylov::gmres_with_info<double>(
    ops, op, x, b,
    {.relative_tolerance = 1e-12,
     .operator_estimates = {
         .norm = Estimate{1, uni20::krylov::GmresEstimateKind::upper_bound},
         .inverse_norm = Estimate{1000, uni20::krylov::GmresEstimateKind::upper_bound}}});
```

Each supplied quantity records `estimate` (the default) or `upper_bound`.
The latter is the application's assertion of a known bound, not a conclusion
drawn by GMRES. Finite operator norms must be nonnegative, inverse norms
positive, and condition numbers at least one. The inverse-related quantities
assume an invertible operator. Values are converted to diagnostic precision;
known upper bounds are rounded upward when necessary to preserve that property.
Invalid or unrepresentable inputs throw before operator applications.

When at least one quantity is supplied, `info.errors` contains the converted
operator information and the available estimates below. Let `r=b-A*x_hat` and
let `x_*` denote the exact solution:

| Field | Formula | Required information |
| --- | --- | --- |
| `normwise_backward_error` | `||r|| / (||A||*||x_hat|| + ||b||)` | Operator norm |
| `absolute_solution_error` | `||A^-1||*||r||` | Inverse norm |
| `relative_solution_error` | `kappa(A)*||r|| / ||b||` | Condition number, or both norms |

The solution-error formulas estimate worst-case upper bounds, rather than
predicting the actual error in the particular residual direction. The relative
quantity refers to `||x_hat-x_*|| / ||x_*||`, not to the norm of the computed
iterate. An explicitly supplied condition number takes precedence over the
product of the norm estimates. A condition number alone does not determine an
absolute error scale. Missing information leaves a field absent; relative
solution error is absent for zero RHS. Zero-over-zero backward error is defined
as zero. Binary scaling avoids premature overflow/underflow in intermediate
products and ratios. An unrepresentable final diagnostic can be infinity or
underflow to zero without invalidating the iterate.

All output error quantities remain **estimates**, even with known operator
bounds. The residual and its norm are computed in finite precision, and the
diagnostic arithmetic is not interval arithmetic. In particular, a computed
zero residual gives zero error estimates without proving an exact solution.
Also, an upper bound on `||A||` enlarges the backward-error denominator, so it
does not produce an upper bound on backward error. These distinctions follow
the usual [linear-system error analysis](https://netlib.org/lapack/lug/node81.html).

No operator applications or vector reductions are added: the solver already
computes the needed norms when validating candidates. Diagnostics are published
with their residual and retained on later numerical failure. Supplying operator
information does not change stopping; applications can use the estimates when
choosing an explicit tolerance, accepting a result, or repeating at higher
precision. A future solver-level solution-error target needs its own explicit
stopping policy and treatment of residual-evaluation uncertainty.

## Mixed-precision refinement

The context names the three vector spaces and their operations explicitly:

```cpp
using namespace uni20::krylov;
GmresRefinementContext context{
    .working = krylov_space<double>(working_ops, x),
    .inner = krylov_space<float>(inner_ops, inner_prototype),
    .residual = krylov_space<long double>(residual_ops, b),
    .inner_operator = inner_operator,
    .residual_operator = residual_operator,
    .preconditioner = preconditioner,
    .transfer = transfer};
auto info = gmres_refine_with_info(context, x, b);
```

For MPFR, pass a trailing finite precision to each `krylov_space<mpreal>` (or
`krylov_space<complex<mpreal>>`). The spaces can have the same C++ vector type
while using different precisions. Contexts borrow their operations and allocation
prototypes, so these must outlive the solve. A prototype supplies structure;
its initial numerical values are not used as a solution or RHS.

The transfer callback is `transfer(output, input, destination_precision)`.
It must preserve the mathematical vector space and its metadata, with explicit
rounding at the destination precision. Required directions are working to
residual, residual to inner, inner to residual, residual to working, and residual
to residual. `DenseHostVectorTransfer` supplies these conversions for host
prototype vectors. Block/device applications provide a transfer appropriate to
their storage; the solver does not flatten or move their vectors implicitly.

Each correction performs the following operations:

1. Evaluate the original `b - A*x` in the residual space.
2. Normalize this residual before converting it to the inner space.
3. Run GMRES from a zero correction with its own tolerance and iteration budget.
4. Convert the normalized correction back to residual precision and rescale it.
5. Convert to working precision, update a candidate, and evaluate its residual.
6. Publish the candidate and its diagnostic together.

Normalization prevents a residual outside the inner scalar's exponent range
from vanishing solely because it was narrowed before scaling. It does not
guarantee that all operator results or corrections are representable.

The default outer relative tolerance is working-precision epsilon, represented
in the residual real type; absolute tolerance is zero. The default limit is ten
corrections. Inner defaults are those of standalone GMRES. All are configurable
through `GmresRefinementOptions`. Many low-precision corrections may be required
for a very high-precision target; the limit is not an accuracy guarantee.

`GmresRefinementInfo` reports outer status, attempted corrections, total inner
Arnoldi steps, residual-operator calls, the most recent inner result, and the
true residual of the last published working iterate. An unsuccessful inner
solve stops the outer loop without applying its partial correction. No retry,
precision escalation or fallback backend is implied.

Outer `options.operator_estimates` and `info.errors` use **residual precision**,
including the norm of the working iterate after its explicit transfer to the
residual space. `options.inner.operator_estimates`, when supplied, belong to the
inner operator and are evaluated independently at inner precision. Outer
conditioning information is not silently reused for the rounded inner problem.

## Relationship to three- and five-precision refinement

The precision roles follow [Carson and Higham (2018)](https://doi.org/10.1137/17M1140819)
and [Amestoy et al. (2024)](https://doi.org/10.1137/23M1549079). Their factor/setup
precision belongs to the preconditioner; outer updates, residual evaluation,
inner arithmetic and preconditioned application need not share precision.

Here the three vector spaces select update, residual and inner arithmetic.
Operator/preconditioner callbacks can evaluate in additional workspaces and
explicitly convert their result back to the inner space. Stored lower-precision
factors remain lower-accuracy factors even if their application uses higher
precision. No LU factorization is required by the generic solver.

The stopping defaults are Uni20 policy, not prescribed constants from those
papers. In particular, the GMRES-IR5 experiments vary inner tolerance, and their
forward/backward-error analysis is not equivalent to relative residual.
`epsilon` here is the spacing above one, not the papers' round-to-nearest unit
roundoff (half that spacing for binary arithmetic). The implementation uses
restarted right-preconditioned GMRES; it does not claim all guarantees derived
for particular paper algorithms/preconditioners merely by exposing their
precision roles.

A matrix-free application alone cannot supply `abs(A)*abs(x)` or an operator
norm. Without optional operator information, the reported residual is neither
backward error nor a solution-error estimate. The supplied-norm diagnostics above
do not provide componentwise backward error. A stricter residual target can be
unattainable even when a backward-error criterion would pass.

## Transfer/environment equations

The tensor application supplies the fixed-point projector or gauge constraint
and a compatible RHS. For example, `I-T+P` is invertible when the fixed-point
projector P removes the nullspace problem of `I-T`. Increasing precision cannot
repair a missing gauge constraint. Nor can arithmetic recover information
already lost in stored or truncated tensors.

`gmres_transfer_example` demonstrates this projected equation and a nonzero
initial guess. `gmres_refinement_example` uses a matrix-free stencil with float
corrections, double updates and long-double residuals, checking against a known
solution. The U(1) regression uses actual BlockTensor operations and checks the
retained quantum-number metadata and block values.

## Further work

### Opt-in norm and condition estimation

The supplied-information interface also accepts results from a future estimator,
whose additional work must be explicit and separately counted. Candidate steps:

1. Estimate the largest singular value using operator and adjoint applications,
   for example through Golub-Kahan bidiagonalization or an appropriate iteration
   on `A^H A`. The vector-space interface would gain an optional adjoint capability;
   the current solve does not require one.
2. Estimate the inverse norm through a smallest-singular-value calculation or
   applications of `A^-1` and `A^-H`. Inexact auxiliary solves need controlled
   tolerances and visible failure status; using a preconditioner alone estimates
   a different operator. Permit a separate precision and work budget.
3. For stronger error guarantees, obtain bounds on residual-evaluation error
   and account for rounding in norms and diagnostics. A more accurate residual
   and a more accurate condition estimate address different uncertainties.

Projected singular values can miss poorly conditioned directions outside the
sampled subspace, especially with restarts. Such results remain estimates unless
an additional argument supplies a bound. For nonnormal transfer operators, an
eigenvalue gap alone does not determine the inverse 2-norm. Expensive estimation
must remain optional; known analytic bounds can be supplied without any of this
work. See [Templates, stopping criteria](https://netlib.org/templates/templates.html)
for the distinction between residual, inverse-norm and backward-error criteria.

Explicit flexible GMRES, recycled subspaces, adaptive precision policies,
automatic norm/condition estimation, and asynchronous execution remain separate
extensions. Device/distributed callbacks may implement the current
synchronous contract, including host scalar reductions, but this implementation
does not certify those execution paths. Future asynchronous solves must use
Uni20's scheduler and keep provider precision scopes within the executing
synchronous provider call.
