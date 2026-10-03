# Numerical precision validation

Numerical support is a claim about an **operation, scalar precision, and execution
backend**. Compiling a scalar or passing tests for another backend is insufficient.
Use the shared cases in [tests/numerics](../../tests/numerics/) when extending
scalar-generic algorithms. Existing algorithm-specific regression suites remain
necessary for layouts, status reporting, restarts, and exceptional inputs.

## Shared cases

`precision_cases.hpp` registers real and complex cases for float32, float64,
float80, float128, and `mpreal` at 128 and 256 bits. Both MPFR cases use the same
C++ scalar type. Values, tensor defaults, epsilon queries, and error comparisons
are constructed in that case's precision. MPFR exact arithmetic has separate
tests: passing an exact rational identity does not demonstrate finite-precision
behavior.

All twelve cases remain registered when optional dependencies are disabled.
Google Test reports missing configurations explicitly. Expected GEMM/solve and
projected LAPACK coverage are declared separately in the test registry, rather
than derived from the production concepts: accidentally removing a supported scalar must fail instead
of silently shortening the tested type list. Update that declaration and the
operation-specific expectations when adding support.

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
| CPU matrix one-norm and tensor reductions | Retained increment, conjugating inner product, squared-norm identity |
| CPU and provider GEMM | Analytic real/complex products with an increment near each precision's resolution |
| CPU and provider square solve | Nearly singular dyadic system with analytic solution, independent scalar residual |
| Solve accuracy | Well-conditioned system with exact rational solution, compared with a 512-bit oracle |
| Extended-range solve | System scaled by `2^-1200`, independent of the small-gap test |
| Projected tridiagonal eigensystem | Analytic closely spaced eigenvalues |
| Lanczos | Closely spaced eigenvalues and analytic componentwise eigenvector residuals |
| Arnoldi | Closely spaced real and complex eigenvalues |
| Hermitian exponential action | Diagonal analytic exponential, retaining small output increments |

The shared gap is `2^(8-p)` for a `p`-bit significand. Tolerances scale with the
case's epsilon and are smaller than the gap. Narrowing controls verify that the
gap disappears in a lower native format for the higher-precision cases.

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

The report distinguishes `passed`, `failed`, `unsupported`, `unavailable`, and
`not_applicable`. Unknown skips and unexecuted entries are errors. A filtered
XML file reports only that subset; use an unfiltered execution for the full
matrix. Missing optional dependencies never count as passes. CI runs the matrix
in ordinary configurations and saves XML/Markdown artifacts in the MPLAPACK job.

## Certification boundaries and remaining work

This is the first shared test matrix, not complete certification of every
numerical subsystem. In the current implementation:

- Float80 CPU arithmetic, reductions, GEMM, and solve have shared precision
  probes. Float80 BLAS/LAPACK provider integration is separate work.
- MPFR/MPC CPU matrix norms and reductions have shared probes. Their CPU GEMM
  kernel accepts exact mode only; finite-precision GEMM and solve use MPLAPACK.
- Float80 and MPFR/MPC projected LAPACK wrappers and end-to-end Krylov solvers
  remain unsupported here. Helper-level runtime epsilon tests do not establish
  Lanczos, Arnoldi, or exponential-action support.
- Shared coverage must expand to decompositions, broader matrix functions,
  iterative stopping/restart criteria, async precision propagation, and other
  numerical subsystems before their scalar support is certified. Existing
  specialized tests supply additional evidence but must not be mistaken for
  completed matrix cells.

New numerical operations should join these cases as they are implemented.
Keep operation-specific unsupported reasons explicit, and replace a skip with
the same mathematical test when that combination becomes supported.
