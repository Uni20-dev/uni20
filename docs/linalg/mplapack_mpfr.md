# MPLAPACK arbitrary-precision matrix products and solves

The optional CPU backend connects `mpreal` and `complex<mpreal>` tensors to
MPLAPACK's serial MPFR/MPC routines. It currently supports GEMM and general square
solves using LU factorization. It does not extend the native LAPACK scalar traits,
Krylov support, other factorization APIs, or GPU arithmetic.

## Configuration

Install the GMP, MPFR and MPC development packages (`libgmp-dev`, `libmpfr-dev`,
`libmpc-dev` on Ubuntu), then configure:

```sh
cmake -S . -B build_codex/mplapack_mpfr \
  -DUNI20_ENABLE_MPFR=ON -DUNI20_ENABLE_MPC=ON \
  -DUNI20_ENABLE_MPLAPACK_MPFR=ON
```

`UNI20_USE_SYSTEM_MPLAPACK=AUTO` prefers an installed MPLAPACK package at least
version 3.0.0 with target `mplapack::mplapack_mpfr`, otherwise fetching pinned
v3.0.0. `ON` requires an installed package; `OFF` fetches the pinned source.
The fetched build uses the reference library, with optimized, CUDA and OpenCL
variants disabled. Provider-internal parallelism is outside this adapter's
precision contract.

This option is independent of `UNI20_ENABLE_MPLAPACK`, which enables binary128.
Both can be enabled together; neither implies the other. Real-only scalar and
tensor support still needs only `UNI20_ENABLE_MPFR`, without MPC or MPLAPACK.
The provider's MPFR component itself needs MPC even for real solves.

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
scanned for consistency. These numerical provider kernels require finite working
precision: all-exact input defaults require an explicit override, unless a
participating GEMM output supplies finite precision. Exact elements and exact
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

`gemm(output, alpha, a, b, beta[, p])` keeps the output's existing shape. With no
override and nonzero `beta`, its default also participates in precision selection;
exact defaults are neutral and differing finite defaults are rejected.
All participating values, including coefficients, convert at the backend boundary.
Coefficients may be exact scalar values, integers or `_mp` literals; for example,
`gemm(output, 1, a, b, 0)` needs no separately constructed scalar constants.
Both synchronous and asynchronous interfaces constrain coefficient sources:
`Precision` and `uninitialized` tags are rejected as coefficients at compile time.
An operation precision belongs in the trailing `p` argument, not in `alpha` or `beta`.
With zero `beta`, old output elements may be unset; with zero `alpha` or zero
inner dimension, input elements are not numerically read. Shape requirements
still apply. Output storage must not overlap either input.

Every operation also accepts `MplapackMpfrBackend{}` as its first argument.
When enabled, the host storage selector includes it automatically. Bare-mdspan
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

For `SolveOptions<mpreal>`, the tolerance is optional. Omission means exact zero
pivots only; an explicitly supplied tolerance must be finite and nonnegative.
The optional field avoids requiring a working precision merely to describe the
default policy. Its value is converted to the operation precision before the
pivot comparison. Scale and magnitudes use `mpreal`, including for complex input.

The LU factorization completes before pivot-threshold inspection. Exact singular
pivots and nonfinite factors take precedence over the small-pivot policy. Input
NaNs or infinities preserve both workspaces. Overflow during precision conversion
or factorization is a nonfinite result. Success guarantees finite output, not
conditioning or accuracy; evaluate a residual separately when needed.

The destructive coefficient workspace contains backend-dependent factorization
data. It is not a public reusable LU object. The adapter's `getrf` and `getrs`
leaves use reusable zero-based pivot vectors internally.

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
This first slice uses routines with no workspace-query calls.

Include the corresponding headers under `<uni20/linalg/async/>` for scheduled
products and solves. `assign_product(output, a, b[, p])`, explicit-coefficient
`assign_product`, `gemm`, and `solve(a, b[, p])` select or apply precision only
after the required inputs are readable. Async conjugation and reshape aliases
inherit their parent's default in that readable epoch, including aliases created
before the parent was constructed. Later parent-default changes are observed by
subsequent operations through the same alias.
Async `add_product` constructs exact unit coefficients and now works for
runtime-precision tensors without specifying coefficient precision.

## Examples and validation

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
