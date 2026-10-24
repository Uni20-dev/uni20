# Typed factorial factory

`uni20::factorial<Real>(n)` constructs a native real result. The integer
argument cannot specify its floating result type, so this is a typed factory,
like the typed constants. `uni20::factorial<Real>(n,p)` also supports generic
precision-aware code: a native result requires its own precision, while
`Real=mpreal` requires explicit finite or exact precision. The existing
`math::factorial(n,p)` MPFR convenience remains available.

## Contract and acceptance

Use the existing checked signed64 order domain. Boolean and floating arguments
are not integer sources; negative orders throw `std::domain_error` and unsigned
values beyond signed64 throw `std::out_of_range`. Native overflow returns
positive infinity. Zero and one factorial are exactly one. Native precision
arguments must equal `native_precision<Real>()`; incompatible or exact precision
throws `std::invalid_argument`. MPFR retains its existing provider and exact
storage behavior, including the cost of constructing a large exact integer.

| Region / configured native formats | Measure and reference | Intended criterion | Guarantee level | Observations |
| --- | --- | --- | --- | --- |
| Every order with a finite factorial | Error in reference-result spacing, independent exact GMP integer | <=1 unit | Empirical target, not universal correct rounding | See measurements below |
| Factorials exactly representable at the result precision | Exact comparison | Exact value | API contract | See measurements below |
| First overflowing order and larger checked orders | Classification and bounded execution | Positive infinity; no loop proportional to a huge input | API contract | See measurements below |
| Zero/one, invalid source/order/precision | Exact value or exception type | Contract above | API contract | See measurements below |
| A controlled inexact factorial | Wider exact reference | At least 16-fold error reduction between adjacent tested precisions | Demonstrated example | See measurements below |

Assume nearest rounding, no reassociation and a correctly fused FMA. Native
evaluation must work with MPFR disabled; GMP/MPFR are test references only.
Validate every finite order rather than sampling a subset, including the
normal/overflow boundary for each exponent range.

## Numerical investigation

Multiply successive integer factors while retaining a low word.
An FMA recovers the rounded product's residual; combine it with the propagated
low word, then renormalize the two-word sum. This avoids a separate rounding
loss at each multiplication accumulating in the final result. There is no
allocation or native-to-MPFR conversion.

Check the order against the native exponent range before looping. Because
`n! >= 2^(n-1)` for n>=1, any n greater than `max_exponent` necessarily
overflows. For smaller orders stop as soon as a product overflows. Thus huge
valid signed64 inputs take bounded work, independently of their magnitude.

## Measurements and validation

GCC 13/glibc 2.39, with an independently accumulated GMP integer and MPFR512
error measurement. The full integer is also rounded directly to each result
precision, avoiding an intermediate-oracle double rounding in that comparison.

| Format | Last finite order | Maximum observed spacing error, rounded up | Worst order |
| --- | ---: | ---: | ---: |
| float32 | 34 | 0.487237 | 19 |
| float64 | 170 | 0.496336 | 143 |
| binary80 | 1754 | 0.499713 | 1105 |
| binary128 | 1754 | 0.5 | 38 |

The scan covers every order from zero through the first overflowing factorial.
All finite results also matched direct correct rounding on this configuration;
that observation is stronger than the one-spacing-unit implementation target,
not a portable correctly-rounded guarantee. Every exactly representable
factorial is preserved. Large valid signed64 inputs return infinity promptly.

The shared probes retain exact integer references and demonstrate at least
sixteen-fold error reduction on controlled inexact factorials as precision
increases. All 26 targeted MPFR-enabled tests passed, including native
factories, MPFR exact/finite factories and existing scalar facade behavior.
A separate MPFR-disabled Release check passed all eight native tests and ran
the explanatory example.

Full unfiltered integration validation also passed, with examples, CLI and
Python bindings enabled:

| Configuration | Passing CTest cases | Runtime skips |
| --- | ---: | ---: |
| GCC13 Debug, MPFR/MPC/MPLAPACK, native128 | 3225 | 1 NUMA topology test |
| GCC13 Release, MPFR disabled, native128 | 2453 | 1 NUMA topology test |
| Clang19 Debug, MPFR/MPC/MPLAPACK, native80, native128 disabled | 2898 | 1 NUMA topology test |

Each configuration also passed the complete numerical precision executable and
coverage report. CUDA runtime and macOS were not exercised. The NUMA skips
reflect this host's topology and are separate from the passing counts.

`factorial_example` explains result-type selection, precision arguments,
overflow and, when enabled, exact versus approximate MPFR results.
