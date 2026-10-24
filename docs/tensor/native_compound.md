# Native compound powers

This is the numerical investigation for native `math::compound(x, n)`, meaning
`(1+x)^n` evaluated from the original floating-point input. The existing MPFR
contract is in [MPFR special functions](mpreal_special_functions.md).

The target is float32, float64, binary80 `long double`, and configured binary128.
The exponent is an integer checked against the signed 64-bit range; `bool` and
floating-point exponents are excluded. Native evaluation must not depend on
MPFR, Boost, a wider native format, or conversion of the exponent to a floating
value. An explicit precision remains meaningful only for MPFR arguments.

## Value contract

The domain is `x >= -1`, even for integer exponents. Inputs below -1 produce
NaN, including when `n == 0`. For other inputs, including NaN, `n == 0` gives 1.
For nonzero n, NaN propagates. At x = -1 the result is positive zero for positive
n and positive infinity for negative n. At positive infinity these results are
reversed. Both signed zeros give 1. These are the existing
[MPFR compound conventions](https://www.mpfr.org/mpfr-current/mpfr.html#Transcendental-Functions).
Range results are positive; negative bases are outside this API.

The implementation targets round-to-nearest with gradual underflow. As with the
other portable native algorithms, it must not be compiled with unsafe
reassociation or flush-to-zero transformations. No global correctly-rounded
claim is made, and floating-point exception flags are not an API guarantee.

## Acceptance table

These targets are set before accepting an implementation. A spacing unit is
`2^(max(e, min_exponent) - digits)` where e is the binary exponent of the
unrounded reference. This becomes an absolute-error measure for subnormals;
it is not a relative-error test near zero. The reference is MPFR at 512 bits,
with selected hard cases repeated at 1024 bits to check oracle stability.

| Region, all four formats | Measure | Intended bound | Guarantee level | Observed maximum |
| --- | --- | --- | --- | --- |
| Finite x > -1, ordinary n | Error against original-input reference, in spacing units | At most 4 | Empirical acceptance target | Within target in scan below |
| Tiny x where native 1+x is 1; exponents through signed64 extrema | Same measure; compare the exact input, not a rounded sum or exponent | At most 4; retain resolvable compounding | Empirical acceptance target | Within target in scan below |
| x immediately above -1; overflow/underflow boundaries; subnormal output | Spacing units, positivity and independently rounded classification | At most 4, correct sampled zero/infinity classification | Empirical acceptance target | Maximum <0.75 units; no sampled range failures |
| Poles, signed zeros, infinities, NaNs, n=0 | Exact values and classifications above | Exact agreement | API contract | Regression tests pass |
| Powers of two with representable exact results | Exact equality | Exact agreement | API contract | Regression tests pass |

Tests must include the immediate floating neighbors of region boundaries,
large exponents with tiny x, negative exponents whose positive power would
already underflow/overflow, and integer exponents not representable in the
input format. Ordinary-domain samples alone cannot certify this function.

## Algorithm and error control

The portable implementation retains `1+x` as an expansion of native words, then performs scaled
binary integer exponentiation with guard words and one final reciprocal for a
negative exponent. Keeping the binary scale separately avoids premature range
failure. The extra words must account for error amplification by an exponent as
large as 2^63; merely using a two-word accumulator for every format is not an
adequate error argument.

The numerical building blocks are TwoSum, expansion accumulation and compression
as described by [Shewchuk, *Adaptive Precision Floating-Point Arithmetic and
Fast Robust Geometric Predicates* (1997)](https://people.eecs.berkeley.edu/~jrs/papers/robustr.pdf),
with an FMA residual for products. The implementation is written independently from
these formulas; no upstream implementation or coefficient table is copied.
This is an internal bounded arithmetic kernel for compound, not a new public
scalar type or arbitrary-precision framework.

The expansion has enough words for at least `digits + 72` significand bits:
four words for binary32, three for binary64/binary80 and two for binary128.
Each product first accumulates the pairwise products and their FMA residuals,
compresses the expansion, then discards only the smallest excess components.
It does not truncate an uncompressed sparse expansion. Storage is bounded and
local; there is no allocation or shared mutable state.

With u = 2^(-digits) and k retained words, the truncation error scale is O(u^k)
per product. Repeated squaring amplifies early relative errors in proportion to
the integer power, so the relevant scale is O(|n| u^k), not merely the number
of loop iterations times u. The 72 extra bits allow for the 63-bit magnitude
and guard margin. This explains the word count; it is not a rigorous uniform
four-unit bound for all representable inputs. The full implementation, including
subnormal components, final reciprocal and rounding, is accepted against the
empirical criteria above.

The leading component is normalized near unity after each product, retaining
an integer binary exponent separately. Intermediate powers therefore cannot
prematurely become zero or infinity. All positive powers of the positive base
lie on the same side of one: once an intermediate exponent exceeds a conservative
bound larger than either native result range, the final classification is known.
That check also bounds the integer scale before its next addition/doubling.
Negative exponents keep the positive power in this scaled representation and
apply a quotient correction before restoring the scale. This preserves a
representable reciprocal even when the positive power is outside native range.
The unsigned exponent magnitude handles INT64_MIN without signed negation.

Final native rounding followed by subnormal scaling can produce double rounding;
the acceptance target allows this. Exact representable powers of two are tested
separately. Tiny internal terms can underflow, but after normalization their
absolute loss is far below the result precision even after signed64 amplification;
subnormal *outputs* are preserved by the separate scale.

## Providers and observations

CMake checks the exact C23 signature `R compoundn(R,long long)` and the linked
symbol separately for float, double, long double and binary128, using the usual
`f`, empty, `l` and `f128` suffixes. A matching provider is preferred. The old
TS signature is not assumed. See the [libc function reference](https://sourceware.org/glibc/manual/latest/html_node/Exponents-and-Logarithms.html).
The configured capability names are
`UNI20_HAS_COMPOUNDN_{FLOAT,DOUBLE,LONG_DOUBLE,FLOAT128}`.

On GCC 13 with glibc 2.39, none of these providers is available. The public and
direct-portable paths therefore exercise the same native algorithm. This is not
numerical validation of a newer libc's provider. Enabling MPFR does not alter the
native path; MPFR is used only by the test oracle.

A deterministic prototype scan evaluated 73,334 candidate inputs per format,
excluding out-of-domain cases: tiny increments, signed64 exponent limits,
ordinary values, neighbors of poles and range boundaries, fixed-seed random
inputs, and deliberately finite nontrivial outputs for exponents of every bit
width. No sampled range-classification failures or errors above the target were
observed. Integrated shared probes additionally repeat selected 512-bit oracle
results at 1024 bits and compare an independent exact rational reference.

| Format | Observed maximum spacing error, rounded upward | Representative worst input and exponent |
| --- | --- | --- |
| float32 | 0.749 | x = -0.4999999105930328369140625, n = 127; subnormal double rounding |
| float64 | 0.501 | x = -2^-18, n = 3 in the prototype |
| binary80 | 0.501 | x = 2^-32, n = 2 in the prototype |
| binary128 | 0.501 | x = -2^-38, n = 3 in the prototype |

These are observed maxima, distinct from the intended four-unit target.
The shared-test XML records each path's actual maximum, input and exponent.
The targeted checks passed 29 tests, including all four native formats, MPFR
at 128/256 bits, and the existing scalar facade tests. The example also ran.
Full unfiltered builds and CTest runs, including examples, CLI and Python
bindings, passed in the following local Linux configurations:

| Configuration | Passed | Skipped |
| --- | ---: | ---: |
| GCC 13 Debug, MPFR/MPC, binary80/128, MPLAPACK | 3160 | 1 NUMA topology test |
| GCC 13 Release, MPFR disabled, binary80/128, MPLAPACK kernels disabled | 2414 | 1 NUMA topology test |
| Clang 19 Debug, MPFR/MPC, binary80, binary128 disabled, MPLAPACK | 2845 | 1 NUMA topology test |

Unfiltered numerical suites and complete coverage reports passed in all three
cells. Clang's configuration does not certify binary128. Actual C23 compound
providers were unavailable on this host; synthetic provider-wiring checks do
not certify their numerical behavior. macOS, CUDA and MPI execution were not
exercised.

## Cost

The loop has at most 64 iterations and bounded work per multiplication. It is
more expensive than ordinary native `pow` or `exp(n*log1p(x))`; those expressions
do not satisfy the original-input accuracy contract in general. There is no
claim of comparable throughput. A typed libc provider, where available, avoids
this portable implementation.

Illustrative timings on an Intel Alder Lake workstation, GCC 13 `-O2`, MPFR
disabled, 4,000 varied inputs per case (microseconds per call):

| Format | x approximately 1/128, n = 360 | x approximately 2^-60, n = 2^60 | x approximately 1/8, n = -13 |
| --- | --- | --- | --- |
| float32 | 6.33 | 35.44 | 1.55 |
| float64 | 1.33 | 11.98 | 0.46 |
| binary80 | 22.94 | 153.82 | 4.86 |
| binary128 | 34.62 | 226.31 | 7.16 |

The same accuracy scan found that `exp(n*log1p(x))` could exceed the four-unit
target by large factors: observed maxima were approximately 117, 1,210, 21,693
and 18,862 units for float32/64/80/128. Reproducible inputs from those cases
are retained in the tests. The guard arithmetic addresses a demonstrated
accuracy difference, not just a theoretical concern.

These are rough local measurements under concurrent build load, not timing
assertions or portable performance guarantees. At the tiny-increment point,
naive `pow(1+x,n)` returns 1 for float32/64. Faster special paths can be added
later if measured application use warrants them, with the same accuracy gates.

Run `native_compound_example` to see the retained-increment and reciprocal-range
cases. Numerical validation is registered through the shared precision matrix;
no-MPFR builds still exercise native value/identity tests and the example.
