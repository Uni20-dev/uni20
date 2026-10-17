# MPLAPACK arbitrary-precision dense linear algebra

The optional CPU provider connects `mpreal` and `complex<mpreal>` tensors to
MPLAPACK's serial MPFR/MPC routines through `BlasBackend` and `LapackBackend`.
It supports real and complex GEMM, square solves, [reusable LU and log determinants](lu.md),
reduced QR/LQ, symmetric/Hermitian eigensystems, SVD, and general eigensystems.
It does not extend the native raw-pointer LAPACK scalar traits, Krylov support,
Schur routines, or GPU arithmetic.

## Configuration

Install the GMP, MPFR and MPC development packages (`libgmp-dev`, `libmpfr-dev`,
`libmpc-dev` on Ubuntu), then configure:

```sh
cmake -S . -B build_codex/mplapack_mpfr \
  -DUNI20_ENABLE_MPREAL=ON \
  -DUNI20_BACKEND_MPLAPACK=ON
```

`UNI20_USE_SYSTEM_MPLAPACK=AUTO` prefers an installed MPLAPACK package at least
version 3.0.0 with target `mplapack::mplapack_mpfr`, otherwise fetching pinned
v3.0.0. `ON` requires an installed package; `OFF` fetches the pinned source.
The fetched build uses the reference library, with optimized, CUDA and OpenCL
variants disabled. Provider-internal parallelism is outside this adapter's
precision contract.

The scalar option enables both `mpreal` and `mpcomplex`, using GMP, MPFR and
MPC. Without `UNI20_BACKEND_MPLAPACK`, scalar arithmetic and tensor storage
remain available without MPLAPACK. With it, the MPFR component is selected
automatically, alongside components for any enabled fixed-precision families.
To request only arbitrary-precision kernels, set `UNI20_ENABLE_FLOAT80=OFF`
and `UNI20_ENABLE_FLOAT128=OFF`.

## Selecting working precision

```cpp
#include <uni20/core/math.hpp>
#include <uni20/linalg/ops/matrix_product.hpp>
#include <uni20/linalg/ops/linear_solve.hpp>

auto p = uni20::Precision::bits(256);
uni20::DenseMatrix<uni20::mpreal> a(2, 2, p), b(2, 1, p), product;
// Fill a and b with numerical values at explicit precision.
uni20::linalg::assign_product(product, a, b); // Matching defaults select p.
auto x = uni20::linalg::solve(a, b);          // Inputs preserved.
auto y = uni20::linalg::solve(a, b, uni20::Precision::bits(400));
```

Without an override, finite input construction defaults must match; an exact
default is neutral. Missing defaults are errors, and element precision is not
scanned for consistency. MPLAPACK requires finite working precision. GEMM at
`Precision::exact()` instead uses the reference CPU kernel: MPLAPACK declines
the instance without reading or modifying elements, and normal dispatch falls
through. Exact participating elements and coefficients then give an exact
rational product. This also works with the scalar family alone, without MPLAPACK.

An all-exact solve requires an explicit finite override. A GEMM output with
nonzero `beta` may supply finite precision through its construction default.
In finite mode, exact elements and exact
coefficients are rounded at that selected precision. `assign_product(output, a, b, p)`
and `solve(a, b, p)` select explicit working precision. A successful owning result
carries that default; writing through a structural view leaves the parent
construction default unchanged.

Precision policy belongs to the tensor operation layer: it selects the working
precision and records defaults on owning outputs. The scalar-specific backend
converts participating values, allocates its numerical temporaries and establishes
provider precision. An explicit `p` is passed directly to that backend, without
changing input defaults to convey the request. Bare descriptors have no tensor
construction defaults, and backend kernels do not mutate owner metadata.

These frontends share their precision interface with fixed-precision scalars.
For example, `solve(a, b, precision_of(a))` and
`assign_product(output, a, b, precision_of(a))` also work for `double`, float80,
float128 and their complex counterparts. A fixed result requires its native
significand size exactly; an incompatible or exact request throws before an
output is resized or overwritten. Fixed kernels receive no extra precision
argument. `Precision` and the generic queries are available without MPFR; see
[generic precision](../tensor/scalar_policy.md#generic-precision).

`gemm(output, alpha, a, b, beta[, p])` keeps the output's existing shape. With no
override and nonzero `beta`, its default also participates in precision selection;
exact defaults are neutral and differing finite defaults are rejected.
In finite mode, participating values, including coefficients, convert at the backend boundary.
Exact mode uses ordinary scalar arithmetic without scanning or reclassifying
elements. Defaults remain best-effort metadata: inserting approximate elements
in a tensor with an exact default can produce approximate results or scalar
precision errors. Selecting exact mode does not restore lost digits.
Coefficients may be exact scalar values, integers or `_mp` literals; for example,
`gemm(output, 1, a, b, 0)` needs no separately constructed scalar constants.
Both synchronous and asynchronous interfaces constrain coefficient sources:
`Precision` and `uninitialized` tags are rejected as coefficients at compile time.
An operation precision belongs in the trailing `p` argument, not in `alpha` or `beta`.
With zero `beta`, old output elements may be unset; with zero `alpha` or zero
inner dimension, input elements are not numerically read. Shape requirements
still apply. Output storage must not overlap either input.

Products accept `BlasBackend{}` as their first argument; square solves and
reusable LU accept `LapackBackend{}`. The host storage selector chooses these
families automatically. The scalar-specific kernels select MPLAPACK and keep
its object representation and thread-local defaults at the provider boundary.
There is no separate MPFR backend selector. `UNI20_BACKEND_BLAS=OFF` disables
native BLAS kernels but does not disable explicitly configured MPLAPACK MPFR
products; the host selector still includes `BlasBackend` for those kernels.

An explicit BLAS-only selector cannot execute exact GEMM; use the ordinary
selector for CPU fallback. Bare-mdspan
kernel dispatch needs an explicit trailing `Precision`, because mdspans do not
carry tensor defaults. Host-readable transforming accessors are packed through
logical element access; a conjugating view is not bypassed through its pointer.

## Solves and diagnostics

The ordinary [square-solve contract](linear_solve.md) applies, including
preserving and destructive interfaces, multiple right-hand sides, empty no-ops,
nonoverlapping workspaces, and numerical outcomes in `SolveInfo`.

```cpp
using namespace uni20::linalg;
auto info = solve_inplace_with_info(a_work, b_work, p,
    SolveOptions<uni20::mpreal>{
        .relative_pivot_tolerance = uni20::mpreal("1e-70", p)});
```

`SolveOptions<mpreal>` uses the same value field as other real types. Its default
is `mpreal{}`, an exact zero requiring no working precision, so the default policy
rejects only exact zero pivots. A supplied tolerance must be finite and
nonnegative. Its value is converted to the operation precision before the
pivot comparison. Scale and magnitudes use `mpreal`, including for complex input.

The LU factorization completes before pivot-threshold inspection. Exact singular
pivots and nonfinite factors take precedence over the small-pivot policy. Input
NaNs or infinities preserve both workspaces. Overflow during precision conversion
or factorization is a nonfinite result. Success guarantees finite output, not
conditioning or accuracy; evaluate a residual separately when needed.

The destructive coefficient workspace contains backend-dependent factorization
data. It is not a public reusable LU object. The adapter's `getrf` and `getrs`
leaves use reusable zero-based pivot vectors internally. The separate
`lu_factor` frontend publishes an owning, read-only factor object for repeated
`lu_solve` and `slogdet` calls at the factor precision.

Successful nonempty in-place solves record the selected working precision on
owning workspaces, whether selected explicitly or inferred from their defaults.
Writing through a view preserves the parent's default. Empty in-place solves
preserve both values and defaults; failed solves do not update defaults, although
factorization may have modified values. A newly returned preserving solution
with explicit precision carries `p` even when empty.

Preserving solves select their default backend from the materialized column-major
host workspace types, consistently for inferred and explicit precision. This also
allows generated inputs such as `solve(full(mpreal{2}, 1, 1),
full(mpreal{6}, 1, 1), p)`. An explicitly supplied backend selector is honored;
an unsupported explicit selection reports dispatch failure.

## Provider context and asynchronous execution

Provider objects have their own representation and destination-preserving
assignment policy. The adapter constructs actual provider arrays, copies through
MPFR/MPC APIs, and returns owning Uni20 values. It never reinterprets Uni20 arrays
as provider arrays and never converts through native floating point. The current
implementation first materializes logical host matrices and then packs provider
arrays; eliminating the extra packing allocation is deferred.

A lower-precision element keeps its existing approximation when copied into a
higher-precision workspace. Higher-precision elements round to the selected
precision. Increasing the operation precision cannot recover input digits that
were already lost.

Each provider invocation creates a synchronous scope on its executing thread.
It establishes both real and complex defaults, nearest-even rounding and the
wrapper's cached rounding state. It accounts for first-use initialization and
restores the prior defaults, override state and exponent bounds on normal and
exceptional exits. No scope crosses an await or a handoff to another thread.
Workspace queries and their subsequent execution run inside the same synchronous
precision scope. Query results are converted directly from the provider's MPFR
representation into checked integer lengths; they never pass through `double`.
All numerical workspace objects are constructed while the scope is active.

Include the corresponding headers under `<uni20/linalg/async/>` for scheduled
products and solves. `assign_product(output, a, b[, p])`, explicit-coefficient
`assign_product`, `gemm`, and `solve(a, b[, p])` select or apply precision only
after the required inputs are readable. Async conjugation and reshape aliases
inherit their parent's default in that readable epoch, including aliases created
before the parent was constructed. Later parent-default changes are observed by
subsequent operations through the same alias.
Async `add_product` constructs exact unit coefficients and now works for
runtime-precision tensors without specifying coefficient precision.

## Dense decompositions

The ordinary tensor APIs support real `mpreal` and `complex<mpreal>` through
`LapackBackend`. Fixed scalar types use the same precision arguments where their
provider is available. Binary80 decompositions are not yet wired.

```cpp
#include <uni20/linalg/ops/qr.hpp>
#include <uni20/linalg/ops/lq.hpp>
#include <uni20/linalg/ops/self_adjoint_eigh.hpp>
#include <uni20/linalg/ops/svd.hpp>

// A is a populated matrix; H is a square symmetric/Hermitian matrix.
auto [q, r] = uni20::linalg::qr(A);              // Infer A's default.
auto [l, q2] = uni20::linalg::lq(A, p);
auto [u, s, vh] = uni20::linalg::svd(A, p);
auto singular = uni20::linalg::singular_values(A, p);
auto [w, v] = uni20::linalg::eigh(H, p);
auto full = uni20::linalg::svd(A, p,
    uni20::linalg::SvdOptions{.left = uni20::linalg::SvdVectorExtent::Full,
                            .right = uni20::linalg::SvdVectorExtent::Full});
```

`svd_left` and `svd_right` also accept `p`, followed by an optional vector extent.
`truncated_svd(A, p[, policy])` applies the existing truncation policy after SVD
and preserves precision defaults on the smaller owning factors. Cutoff values
obey the scalar arithmetic rules: exact thresholds are precision-neutral;
approximate thresholds used in arithmetic must match the factor precision.
Empty or identically zero diagnostics can remain exact zero.
The explicit-precision form of `eigh` accepts an optional triangle after `p`.
Existing settings-only calls retain their behavior. An explicit selector, when
used, is always the first argument. Destructive `qr_factorization`,
`lq_factorization`, `self_adjoint_eigh` and `singular_value_decomposition` expose
the same precision policy; for APIs with settings, `p` precedes those settings.
The fixed-output `nonsymmetric_eigen` takes `p` after its `compute_right_vectors`
flag and returns complex eigenvalues/vectors for either real or complex input.

These algorithms require finite working precision, even for exact input or an
empty matrix. An exact or absent input default requires an explicit override.
Only the matrix input supplies an inferred default: output defaults do not
participate. Exact input elements are rounded at the provider boundary. Owning
results and successful destructive workspaces record the selected precision;
structural output views retain their parent's default. Synchronous calls reject
fixed-type precision mismatches before consuming input or resizing outputs.

QR/LQ produce reduced factors with orthogonal or unitary Q. Self-adjoint
factorization reads only the selected triangle. SVD returns nonnegative singular
values in descending order and `Vh`, the adjoint of V. Its full empty factors
are identity matrices. General eigenvalues are unsorted; real conjugate pairs
are unpacked into actual complex vector columns. Provider convergence failures
use the same `LapackError` and error policy as native decompositions. Workspaces
are not rollback buffers after a numerical failure.

Preserving calls leave inputs unchanged. Consuming calls keep the existing
storage-transfer rules, including reduced SVD factors adopting input storage and
row-major self-adjoint eigensystems reusing storage with the appropriate triangle
and conjugation adjustments. Provider packing remains necessary even when the
Uni20 output adopts input storage.

Include the matching headers under `linalg/async/` for preserving and consuming
QR/LQ, self-adjoint eigensystems and all SVD forms. Precision inference happens
after the input becomes readable; an explicit request is copied into the
coroutine frame. General nonsymmetric eigenanalysis retains its synchronous
fixed-output API; no new async or preserving interface is implied.

Async calls validate precision inside the scheduled task. A precision failure
leaves a preserving call's input readable and propagates to every result epoch.
A consuming call has enrolled its input as a writer, so the same failure also
propagates to the input epoch; it does not guarantee preservation of the input.
This follows the ordinary [async failure contract](../tensor/operations.md#failures),
including when an explicit request could be rejected without reading elements.

## Examples and validation

- `dense_decomposition_precision_example`: QR/LQ, SVD and eigensystems of one matrix
  at native and runtime precision, with labeled reconstruction errors.
- `generic_precision_example`: one square-solve interface for native and MPFR scalars.
- `tensor_precision_example`: construction defaults, inherited view defaults and bulk conversion.
- `mplapack_mpfr_example`: a complex 400-bit solve which becomes singular at
  113 bits, with a separately evaluated 512-bit residual.
- `mplapack_precision_example`: concurrent async solves at four independent precisions.

The MPFR backend tests cover real/complex products and LU solves, exact values
beyond binary128, rounding at the packing boundary, singular pivots, context
restoration and exceptional exits. Tensor-level tests cover logical conjugation,
precision selection, solve diagnostics, independently evaluated residuals and
consumers scheduled before their inputs are published. Fixed-precision tests
remain separate regressions; enabling MPFR does not change their arithmetic.
Dispatch tests verify BLAS/LAPACK family selection, explicit async selectors,
exact GEMM fallback, and type-level rejection when the provider or requested
operation is unavailable. The shared numerical cases cover real and complex
float32, float64, configured binary80/binary128, MPFR-128 and MPFR-256
decompositions (subject to the binary80 complex-provider restriction). They check
reconstruction, orthogonality, eigenpair residuals, small spectral gaps and
accuracy against an independent 512-bit reference. Separate API tests cover
storage transfer, exact/missing defaults, selected-triangle reads, empty factors,
logical conjugation, provider-context restoration and asynchronous precision.

## Integration roadmap

The shared precision API and backend-family selection are implemented. The
remaining work extends routine coverage without introducing scalar-specific
public frontends or weakening the existing tensor and async contracts.

1. **Completed: shared precision and backend selection.** Fixed and runtime
   scalar types use the same precision arguments. MPFR GEMM specializes
   `BlasBackend`; MPFR square solves and reusable LU specialize `LapackBackend`.
   Logical host packing is shared between these kernels. Provider arrays,
   integer lowering, conversion, and synchronous precision scopes remain in
   `backend/mplapack/`.
2. **Completed: binary80 family integration and dense capability boundaries.**
   Binary80 GEMM uses `BlasBackend`; its solves, LU and decompositions use
   `LapackBackend`. Packed MPFR and binary80 decomposition kernels share logical
   packing and provider algorithms, with per-operation capability checks.
   Raw-pointer wrappers retain their narrow `BlasScalar`/`LapackScalar` gates;
   partial packed-provider coverage does not admit unrelated raw routines or
   projected algorithms. Binary80 complex LU/decompositions remain disabled for
   providers with unsafe `_Float64x` complex division. Continue replacing broad
   scalar requirements with actual operation requirements when extending the
   projected algorithms below.
3. **Completed: dense decomposition batch.** Real/complex QR and LQ, symmetric/
   Hermitian eigensystems, all SVD output forms and general eigensystems share
   the precision interface and provider workspace handling. Native complex
   QR/LQ wrappers are included. The provider routines are `R/Cgeqrf`,
   `Rorgqr/Cungqr`, `R/Cgelqf`, `Rorglq/Cunglq`, `Rsyev/Cheev`, `R/Cgesvd`
   and `R/Cgeev`.
4. **Remaining dense capabilities.** Add tridiagonal eigensolvers, Schur
   decomposition/reordering and any further routines required by projected
   algorithms. Their precision and scalar eligibility need explicit coverage;
   dense QR/SVD support does not automatically enable them.
5. **Projected algorithms and allocation costs.** Enable MPFR Krylov paths only
   once their required dense routines, constants, tolerances and workspaces are
   precision-generic and validated. Remove redundant host/provider packing where
   measurements justify it; preserve logical accessor semantics and the owning
   factorization contract. Provider-internal threading and GPU execution require
   their own contracts and are outside the current serial CPU adapter.

MPLAPACK's upstream routine inventory is broader than Uni20's wrappers.
The [dense coverage inventory](dense_blas_lapack_coverage.md) records the
implemented boundary separately from upstream availability.
