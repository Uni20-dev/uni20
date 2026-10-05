# Numerical precision validation

Numerical support is a claim about an **operation, scalar precision, and execution
backend**. Compiling a scalar or passing tests for another backend is insufficient.
Use the shared cases in [tests/numerics](../../tests/numerics/) when extending
scalar-generic algorithms. Existing algorithm-specific regression suites remain
necessary for layouts, status reporting, restarts, and exceptional inputs.

## Shared cases

`precision_cases.hpp` describes real and complex cases for float32, float64,
float80, float128, and `mpreal` at 128 and 256 bits. Both MPFR cases use the same
C++ scalar type. Values, tensor defaults, epsilon queries, and error comparisons
are constructed in that case's precision. MPFR exact arithmetic has separate
tests: passing an exact rational identity does not demonstrate finite-precision
behavior.

All twelve cases remain in the coverage matrix when optional dependencies are
disabled. `precision_registry.hpp` declares which operation/scalar/backend
combinations are configured and supported. Only those combinations become
executable Google Tests. Unsupported, unavailable, and mathematically
inapplicable combinations appear in the coverage report, without creating
skipped tests or counting as passes.

The expectations are independent of production concepts: accidentally removing
a supported implementation must fail compilation or its numerical test instead
of shortening the tested type list. `NumericalCoverage.RegisteredProbesMatchDeclaredMatrix`
also checks that every expected probe is registered exactly once and exports the
full declared matrix into the test XML. Update the registry and operation-specific
expectations when adding support. A fully configured build should run all its
registered tests; reserve runtime skips for conditions discovered during execution,
such as a multi-NUMA-node test on a machine with only one visible NUMA node.

Float128 means a 113-bit significand, not MPLAPACK's fallback alias to a narrower
`long double`. That alias mode is reported as unavailable for this case.
Float80 similarly requires Uni20's native 64-bit-significand format detection.

## Evidence required for numerical support

1. **Correctness:** run the same deterministic mathematical problem across the
   case list, using exact analytic data or an independent higher-precision oracle.
   Include nonzero imaginary components in complex tests.
2. **Retained precision:** use inputs and outputs whose distinguishing increments
   disappear at a lower precision. Tests must reject narrowing in input loading,
   workspaces, tolerances, kernels, and result construction where those paths are
   exercised.
3. **Improved accuracy:** use a controlled problem where additional precision
   lowers a measured error. Do not demand monotonic improvement on arbitrary
   iterative problems or compare two implementations sharing the same oracle.
4. **Exponent range:** test this separately from significand precision. A value
   outside double's range says nothing about retention of float80's low bits.
5. **Runtime precision:** run the same algorithm at multiple MPFR precisions and
   inspect the output precision. Exact and unspecified states need their own
   boundary tests; they cannot supply an implicit epsilon.
6. **Execution coverage:** force the backend when testing a provider. A successful
   fallback does not certify that provider. Async, device, view, and distributed
   paths require additional tests as they gain support.

Avoid `EXPECT_NEAR` for scalar-generic numerical assertions: its arguments are
`double`. The shared `expect_error_at_most` computes the error in the tested real
type and formats only diagnostics. Construct decimals from text or exact
integer ratios; casting a double literal does not create higher-precision data.

## Choosing a numerical assertion

Use exact equality when the contract demands it, including provider-wiring
checks against the same correctly rounded operation. Use an explicit
absolute/relative bound for residuals, accumulated errors and identities near
zero. `EXPECT_FLOATING_EQ` and `ASSERT_FLOATING_EQ` from
`<uni20/common/gtest.hpp>` instead count representable steps between results;
their default tolerance is four ULPs. They support native float32/64/80/128,
native complex values componentwise, and `mpreal` when MPFR is enabled.
`complex<mpreal>` ULP assertions are not yet supported.

For `mpreal`, the comparison contract is:

- Two approximate operands must have the same precision, even if their values
  are equal. A mismatch fails with a diagnostic; convert a reference explicitly
  with `.at(p)` when that is intended.
- An exact operand is rounded to the approximate operand's precision, nearest
  with ties to even, before counting steps. This includes MPFR's ordinary
  underflow/overflow behavior. Two exact operands require exact equality;
  no tolerance gives unequal rationals a floating-point grid.
- The grid uses that precision and MPFR's current exponent range, without
  changing either. MPFR has no subnormals: zero and the smallest positive
  representable value are one step apart. Both signed zeros occupy the same
  position. Finite operands outside the current exponent range fail explicitly.
- NaNs and unset values fail. Equal infinities pass at matching precision;
  other comparisons involving infinity fail. Negative tolerances fail.
- Diagnostic distances saturate at `max<long long>` in magnitude. The actual
  comparison uses the full integer distance, so saturation cannot turn a
  failure into a pass. Exact inequality and incomparable pairs have no distance
  and use the same sentinel, with an explanatory assertion message.

```cpp
auto p = uni20::Precision::bits(256);
auto actual = uni20::mpreal(1, p) / 3;
auto reference = uni20::mpreal("1/3", uni20::Precision::bits(512));
EXPECT_EQ(actual.precision(), p);
EXPECT_FLOATING_EQ(actual, reference.at(p), 1);
EXPECT_FLOATING_EQ(actual, uni20::mpreal("1/3", uni20::Precision::exact()), 1);
EXPECT_FLOATING_EQ(uni20::mpreal(1, p), uni20::mpreal{1}, 0);
```

Operand expressions and the tolerance are evaluated once. Failure messages
show round-trip MPFR values, exact/unset states, working precisions and the
reason an invalid pair cannot be compared. Comparison preserves the operands
and MPFR precision defaults, exponent range and exception flags. The shared
`check::FloatingULP<mpreal>` comparator also serves `CHECK_FLOATING_EQ` and
`PRECONDITION_FLOATING_EQ`; the detailed ULP diagnostic is supplied by the
GoogleTest assertions. ULP proximity does not check the requested output
precision or establish the independence of a reference, so keep those checks
separate.

## First operation families

Scalar math dispatch uses the same real/complex precision cases. A near-one
square-root problem and a magnitude retaining an increment beyond the preceding
precision detect narrowing in the callable interface. Core tests separately
exercise ADL lookup, unsupported types and exact/explicit-precision semantics;
these mechanism tests do not replace the numerical precision cases.

The `uni20_numerical_precision_tests` target currently covers:

| Family | Evidence |
| --- | --- |
| Scalar epsilon and arithmetic | Runtime epsilon, retained increment, complex reciprocal accuracy |
| Real elementary math | Small-argument `expm1`/`log1p` with cancellation negative controls; inverse identities retaining increments; error improvement against independent 512-bit series |
| Real numerical utilities | Named decompositions and remainders retaining increments; fused cancellation; rounding ties, signed zeros, NaN/infinity and grid endpoints; integer power/root accuracy against rational and independent Newton references |
| Native/MPFR Gamma and error functions | Accuracy improvement against `sqrt(pi)` and an independent 512-bit error-function series |
| Native/MPFR Beta, zeta and Ei | Rational and series identities without MPFR; increased accuracy against pi identities and an independent 512-bit Ei series |
| Typed constants | Type retention and improved accuracy of pi, log(2) and Euler gamma, including binary128 constants |
| MPFR special functions | Gamma/digamma/Beta recurrences, upper-Gamma identity, Bessel Wronskian, zeta/pi identity; accuracy against Ei, Li2, Bessel J and Airy series and an AGM iteration |
| MPFR elementary extensions | Small-argument cancellation controls, large pi-scaled argument reduction, inverse and reciprocal identities, compound rounding without loss in `1+x` |
| CPU matrix one-norm and tensor reductions | Retained increment, conjugating inner product, squared-norm identity |
| CPU and provider GEMM | Analytic real/complex products with an increment near each precision's resolution |
| CPU and provider square solve | Nearly singular dyadic system with analytic solution, independent scalar residual |
| Solve accuracy | Well-conditioned system with exact rational solution, compared with a 512-bit oracle |
| CPU and provider reusable LU | Small-gap system, multiple RHS columns and repeated solves; rational solution against a 512-bit oracle |
| CPU and provider log determinant | Near-one magnitude with an independent analytic logarithm bound; nontrivial complex phase; extended exponent range |
| Extended-range solve | System scaled by `2^-1200`, independent of the small-gap test |
| Projected tridiagonal eigensystem | Analytic closely spaced eigenvalues |
| Lanczos | Closely spaced eigenvalues and analytic componentwise eigenvector residuals |
| Arnoldi | Closely spaced real and complex eigenvalues |
| Hermitian exponential action | Diagonal analytic exponential, retaining small output increments |

The shared gap is `2^(8-p)` for a `p`-bit significand. Tolerances scale with the
case's epsilon and are smaller than the gap. Narrowing controls verify that the
gap disappears in a lower native format for the higher-precision cases.
Log-determinant range probes additionally factor a non-diagonal matrix scaled
by `2^±1200` and `2^±12000`: its exact LU factors remain representable, but
unscaled complex division can overflow or underflow internally. This exercises
elimination as well as the logarithmic reduction.

For the non-dyadic reciprocal and solve fixtures, the oracle transfers each
computed binary value exactly to 512-bit MPFR. It compares the error with both
the case's epsilon and the error from rounding the analytic solution at the
preceding precision (12, 24, 53, 64, 113, or 128 bits as appropriate). The result
must be at least sixteen times more accurate than that lower-precision value.
The 12-bit case is a comparison baseline, not another supported Uni20 scalar.
Without MPFR, these oracle-dependent checks are explicitly unavailable; the
analytic increment, range, and residual checks still run.

This matrix generalizes the former binary128-only one-norm, reduction,
extended-range solve, tridiagonal, Lanczos, complex Arnoldi, and Hermitian
exponential probes. Specialized binary128 Schur, triangular Arnoldi, matrix
exponential, SVD, and QR/LQ regressions remain in their existing suites.

## Running and reporting

Use the normal project configuration, adding MPFR/MPC and MPLAPACK when testing
those providers. See [testing](testing.md), [MPFR scalars](../tensor/mpreal.md), and
[MPLAPACK setup](../linalg/mplapack_binary128.md) for configuration details.

```bash
cmake --build build_codex/<configuration> --target uni20_numerical_precision_tests
ctest --test-dir build_codex/<configuration> --output-on-failure -R '^Numerical'
```

For an artifact that records actual execution results, run the executable with
Google Test XML output, then render that XML:

```bash
build_codex/<configuration>/tests/numerics/uni20_numerical_precision_tests \
  --gtest_output=xml:build_codex/<configuration>/precision.xml
python3 scripts/report-numerical-precision.py \
  build_codex/<configuration>/precision.xml \
  --output build_codex/<configuration>/precision.md
```

The report joins the declared matrix to actual results and distinguishes
`passed`, `failed`, `unsupported`, `unavailable`, and `not_applicable`. A missing,
unexecuted, or skipped expected probe is an error, as is an undeclared probe.
Use an unfiltered execution: a filtered XML file cannot stand in for complete
coverage. Missing optional dependencies never count as passes. The coverage
report retains the unsupported combinations even in a build with no skipped
tests. CI runs the matrix in ordinary configurations and saves XML/Markdown
artifacts in the MPLAPACK job.

## Certification boundaries and remaining work

This is the first shared test matrix, not complete certification of every
numerical subsystem. In the current implementation:

- Float80 CPU arithmetic, reductions, GEMM, solve, reusable LU and log
  determinants have shared probes. The optional binary80 provider runs the same
  GEMM, solve and LU/log-determinant probes. Its complex LU/solve cells are
  explicitly unsupported when the provider uses distinct `_Float64x`, whose
  generic complex division can silently overflow; default dispatch uses the CPU
  implementation, but that fallback does not count as provider validation.
- MPFR/MPC CPU matrix norms and reductions have shared probes. Their CPU GEMM
  kernel accepts exact mode only; finite-precision GEMM, solve and reusable
  LU/log determinants use MPLAPACK. LU probes inspect factor, solution and
  determinant precision at both 128 and 256 bits.
- Float80 and MPFR/MPC projected LAPACK wrappers and end-to-end Krylov solvers
  remain unsupported here. Helper-level runtime epsilon tests do not establish
  Lanczos, Arnoldi, or exponential-action support.
- Shared coverage must expand to decompositions, broader matrix functions,
  iterative stopping/restart criteria, async precision propagation, and other
  numerical subsystems before their scalar support is certified. Existing
  specialized tests supply additional evidence but must not be mistaken for
  completed matrix cells.

New numerical operations should join these cases as they are implemented.
The dedicated MPFR-only special/extension probes run at 128 and 256 bits.
Their remaining native implementations are deferred and reported as unsupported,
even when MPFR is enabled. The native Gamma/error-function probes run at all
configured real precisions; their high-precision oracle requires MPFR. Beta,
zeta and Ei have separate native/MPFR probes; their native binary128 cells
remain unsupported. Constant probes also cover configured binary128.

Keep operation-specific unsupported reasons explicit in the registry. When a
combination gains support, enable the same mathematical probe for it.
