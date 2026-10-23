# Native reciprocal functions and Catalan constant

This batch covers native `math::sec`, `csc`, `cot`, `sech`, `csch`, `coth`,
and `catalan<Real>`. AGM and factorial remain separate investigations. The
existing MPFR overloads and their explicit-precision contracts are unchanged.
Native inputs must stay native, including configured binary80 and binary128.

## Contracts and numerical acceptance

Trigonometric reciprocals evaluate at the original represented input. The
rounded native value of pi/2 is not a mathematical pole, so tests must compare
against that input rather than expect an artificial infinity there. Native
sine/cosine/tangent supply argument reduction; this batch adds no phase
approximation. Hyperbolic reciprocal tails must remain nonzero whenever the
rounded result is representable, even if native sinh/cosh would overflow.

| Region / all configured native formats | Measure | Intended criterion | Guarantee level | Observations |
| --- | --- | --- | --- | --- |
| Ordinary finite arguments | Absolute error in reference-result spacing, MPFR512 oracle | <=4 units | Empirical target | See measurements below |
| Near trigonometric poles/zeros; very large arguments | Same exact-input reference measure; distinguish input conditioning from evaluation error | <=4 units | Empirical target, no uniform relative claim about an unrounded argument | See measurements below |
| Hyperbolic tails, minimum normal/subnormal results and range boundaries | Absolute error with subnormal spacing clamped; rounded value classification | <=4 units and matching sampled zero/infinity classification | Empirical target | See measurements below |
| Signed zeros, infinities, NaNs | Exact value/sign/classification | Mathematical reciprocal convention | API contract | See measurements below |
| Native Catalan constant | Compare with independently rounded wider reference | Correct rounding at each configured native format | Specific constant guarantee, not a general function guarantee | See measurements below |
| Controlled independent values | Error compared with the preceding precision | At least 16-fold improvement | Demonstrated examples | See measurements below |

A spacing unit is `2^(max(e,min_exponent)-digits)` for native results, with e
the unrounded reference's binary exponent. Computation assumes nearest rounding
and gradual underflow. Tests cover both sides of every algorithm join.

At signed zero, sec/sech return 1; csc/cot/csch/coth return signed infinity.
At infinity the trigonometric reciprocals return NaN, sech returns positive
zero, csch returns signed zero, and coth returns the sign of its argument.
NaN propagates through all six functions. No errno or exception-flag promise is
added. Existing MPFR exact-state rules remain in its scalar layer.

## Formulas and range handling

Sec, csc and cot use `1/cos(x)`, `1/sin(x)` and `1/tan(x)`. Coth uses
`1/tanh(x)` without an overflowing intermediate. These compositions are
validated against the acceptance criteria; no
universal correctly-rounded claim is made.

For sech/csch, use the native cosh/sinh reciprocal while `abs(x) <= digits`.
The providers remain well inside their exponent range there. Beyond that join,
the denominator correction in `2*exp(-abs(x))/(1 +/- exp(-2*abs(x)))` is smaller
than epsilon squared, so evaluate the common tail `2*exp(-abs(x))`.
Use that direct form while `abs(x) < -min_exponent*log_two<Real>`; the exponential
is still normal with a conservative margin. Farther out, compute
`h = exp(-abs(x)/2)` and `h*(2*h)`. This preserves final subnormals even when
exp(-abs(x)) itself would prematurely round to zero. Restore the sign for csch.
The threshold chooses a formula, not the expected range classification.

An initial half-exponential formula with its full denominator immediately above
abs(x)=1 reached 4.107 spacing units for binary80 near
x=-1.00000000000000000477048955893621950963279232. The wider direct-provider
region avoids amplifying that exponential rounding error. That original input
and both neighbors of every selected join are regression tests.

Catalan is assembled at compile time from a sufficiently long binary
prefix expressed as exact integer chunks and powers of two. Tests verify the
final native rounding against a wider oracle and use the alternating series
sum (-1)^k/(2k+1)^2, accelerated by
[Euler's transformation](https://dlmf.nist.gov/3.9.ii), as an independent test
reference. MPFR's constant routine generates/checks the prefix offline; it is
not a native runtime dependency.

The prefix is `0xea7cb89f409ae845215822e37d32d0c6 * 2^-128`, split into
64-bit integer chunks. Multiplication by binary scale factors and the final
native sum avoid a double intermediate. At 1024-bit reference precision, 400
terms of the Euler-transformed alternating series give an independent check;
the positive transformed coefficients are at most one, so its omitted tail
is bounded by 2^-400. The final constants match the correctly rounded wider
values for all four native formats.

## Observations and validation

GCC 13/glibc 2.39, 512-bit MPFR oracle: roughly 39,000--41,000 deterministic
inputs per format, including ordinary grids, tiny inputs, pole/join neighbors,
full exponent ranges and fixed-seed random significands. Observed maxima in
spacing units, rounded upward:

| Format | sec | csc | cot | sech | csch | coth |
| --- | --- | --- | --- | --- | --- | --- |
| float32 | 1.431 | 1.446 | 1.453 | 2.072 | 2.206 | 2.687 |
| float64 | 1.394 | 1.407 | 1.428 | 2.323 | 2.080 | 1.889 |
| binary80 | 2.038 | 1.856 | 1.580 | 2.560 | 2.560 | 1.702 |
| binary128 | 2.165 | 2.013 | 1.522 | 2.016 | 1.953 | 1.935 |

No sampled range-classification failures remained. These are measurements,
not a replacement for the intended four-unit criterion or a global proof.
The integrated probes record their own maxima and worst inputs in test XML.
Independent sine/cosine/sinh/cosh series check precision improvement at 1/4;
identities and core boundary cases also run with MPFR disabled.

Targeted tests passed: 70 tests including reciprocal functions, constants,
existing scalar/special-function cases and an oracle regression. The oracle's
binary-value reconstruction now preserves negative zero and nonfinite values;
previously it discarded the sign of zero. A separate MPFR-disabled Release
check passed all 12 native reciprocal/constant tests, and the example ran in
that configuration. Full validation is pending.

Run `native_reciprocal_math_example` to see the retained hyperbolic tails and
native Catalan constants. No external dependency or provider selection changes
are introduced by this batch.
