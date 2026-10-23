# Native arithmetic-geometric mean

This implementation extends `math::agm(a,b)` to equal native real types,
including configured binary80 and binary128. MPFR supplies an independent
oracle, not the native implementation. Existing MPFR overloads are unchanged.

## Domain and acceptance criteria

For positive finite arguments, AGM is the common limit of successive
arithmetic and geometric means. It is symmetric, homogeneous and lies between
its arguments. Preserve the [MPFR value conventions](https://www.mpfr.org/mpfr-current/mpfr.html#Transcendental-Functions):
NaNs propagate; zero with a finite argument, even a negative one, gives +0;
zero with infinity gives NaN; other negative arguments give NaN; positive
infinity with a positive argument gives positive infinity. Equal positive
arguments return that value exactly. Negative zero is treated as zero.

| Region / configured native formats | Measure and reference | Intended criterion | Guarantee level | Observations |
| --- | --- | --- | --- | --- |
| Positive ordinary inputs and near equality | Absolute error in reference-result spacing, MPFR512 | <=8 units | Empirical target, allows accumulated rounding in the iteration | See measurements below |
| Extreme operand ratios, normal/subnormal boundaries | Same exact-input measure, subnormal spacing clamped | <=8 units; finite nonzero result between inputs | Empirical accuracy target and range contract | See measurements below |
| Equal inputs, zero, infinity, NaN, negative inputs | Exact value/sign/classification | Conventions above, symmetry | API contract | See measurements below |
| Independent elliptic-integral series | Wider reference and adjacent-precision comparison | At least 16-fold accuracy improvement | Demonstrated controlled example | See measurements below |

A spacing unit is `2^(max(e,min_exponent)-digits)`, where e is the binary
exponent of the unrounded result. Use the original exactly represented
arguments for the oracle. Assume nearest rounding and gradual underflow.
Repeat difficult oracle cases at 1024 bits. These targets precede prototype
evaluation; observed maxima are recorded separately, not used as tolerances.

## Algorithm investigation

The classical iteration and its quadratic convergence are described in
[DLMF 19.8(i)](https://dlmf.nist.gov/19.8.i). No upstream implementation or
coefficient table is adapted. The principal implementation problem is range:
`sqrt(a*b)` can overflow or underflow although the mean is representable.
Scaling both operands by the larger one's exponent can instead erase the
smaller input. AGM(minimum-subnormal,maximum) is a useful counterexample.

Decompose both positive finite inputs as `a*2^ae` and `b*2^be`, with
significands in [1/2,1) and the larger original value first. While
`ae-be > digits+2`, form the geometric mean from the bounded significand
product and the halved exponent sum, including a factor of two for an odd
sum. Normalize that significand without restoring the original exponent.
The arithmetic mean is the larger value divided by two: the omitted smaller
term has relative size below `2^(-digits-2)`, one quarter of unit roundoff.
The exponent gap approximately halves on each iteration.

Once the gap is at most `digits+2`, use the larger exponent as a common scale.
Both scaled inputs and their product are normal in every supported native
format. Iterate `(a+b)/2` and `sqrt(a*b)` until their separation is at most
`epsilon*a`. Return their midpoint and restore the exponent once, permitting
only the final result to become subnormal. Equal original inputs bypass the
iteration. No allocation, wider scalar type or coefficient table is needed.

The limit `2*bit_width(digits)+8` for the common-scale phase is conservative
relative to quadratic convergence from this bounded ratio. Failure to reach
the spacing test returns NaN rather than an unfinished approximation; no such
failure was observed. Accumulated arithmetic rounding and the omitted small
terms are included in the empirical eight-spacing-unit target; this is not a
proof of a uniform correctly-rounded result. The mean stays representable for
positive finite inputs even at the most extreme ratios.

Independent validation uses the complete elliptic-integral identity
`AGM(1,sqrt(1-k^2)) = pi/(2*K(k))`, with K evaluated by its convergent power
series at a modest k. This avoids reusing AGM iteration for both sides.

## Measurements and validation

GCC 13/glibc 2.39; deterministic grid, near-equality neighbors, subnormal
pairs, extreme ratios and fixed-seed random significands/exponents. The oracle
is MPFR512 at the original represented inputs. Repeating the worst sampled
pairs at 1024 bits agrees after rounding to 512 bits.

| Format | Prototype pairs | Observed maximum spacing error, rounded up | Maximum total iterations |
| --- | ---: | ---: | ---: |
| float32 | 30,550 | 2.657 | 11 |
| float64 | 30,904 | 3.541 | 15 |
| binary80 | 36,850 | 3.281 | 19 |
| binary128 | 36,862 | 4.374 | 20 |

No sampled result escaped the input bounds, became zero for two positive
inputs, or encountered an intermediate range failure. These observations do
not replace the stated acceptance targets. The shared tests record their own
maximum errors and worst input pairs as XML properties.

For the independent reference choose a=1, b=1/2, so k^2=3/4. The pi factors
cancel: `AGM(1,1/2) = 1 / sum_n [(binomial(2n,n)/4^n)^2*(3/4)^n]`.
All coefficient squares are at most one, so the tail after 2000 terms is
less than `4*(3/4)^2000 < 2^-828`. Evaluate at 1024 bits, then round to 512.
This agrees with the MPFR AGM oracle and demonstrates at least sixteen-fold
accuracy improvement between the tested precisions.

The 25 targeted tests passed (native contracts, shared native/MPFR precision
probes and existing scalar facade tests). A separate MPFR-disabled Release
check passed all eight native contract tests; the explanatory example also
ran without MPFR. These are iteration checks against compatible library
builds, not a substitute for rebuilding the whole configuration. Full
integration validation is pending.

Run `native_agm_example` to see native precision improvement and a finite
AGM for the minimum-subnormal/maximum double pair. In that case scaling both
original inputs by the larger exponent would incorrectly erase one of them.
