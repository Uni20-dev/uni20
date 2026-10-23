# Native digamma

This implementation adds `math::digamma(x)` at each configured native real
precision. Native arguments stay native; the existing MPFR implementation and
precision contracts remain unchanged. There is no Boost dependency or new
complex function surface.

## Value contract and numerical acceptance

The real digamma is the logarithmic derivative of Gamma. Use MPFR's value
conventions: negative integer poles give NaN, +0 gives -infinity, -0 gives
+infinity, +infinity gives +infinity, and -infinity/NaN give NaN. Finite
negative nonintegers are supported by reflection. Overflow near zero returns
the corresponding signed infinity. Native calls have one argument; no
implicit scalar conversion or native trailing precision is introduced.

The error measure uses an MPFR512 reference at the original represented input,
not at a nearby intended decimal value. Let eps be the native spacing at one.

| Domain / all configured native formats | Measure | Intended bound | Guarantee level | Observations |
| --- | --- | --- | --- | --- |
| Positive finite inputs, including the positive zero neighborhood | `abs(error)/(eps*max(1,abs(psi(x))))` | <=8 | Empirical target; absolute near zero, relative away from zero | See measurements below |
| Negative nonintegers, including reflection cancellation near zeros | `abs(error)/(eps*max(1,log(1+abs(x)),abs(psi(x))))` | <=8 | Empirical target; logarithmic absolute scale near negative zeros | See measurements below |
| Pole neighbors and reciprocal overflow boundary | Above measure for finite values; exact sign/classification at overflow | <=8 and matching sampled range classification | Empirical accuracy target plus range contract | See measurements below |
| Zeros, infinities, negative integer poles, NaNs | Exact value/sign/classification | Conventions above | API contract | See measurements below |
| Controlled integer/half-integer identities | Independent constants/harmonic sums; adjacent precision comparison | At least sixteen-fold improvement | Demonstrated examples | See measurements below |

Near zeros a uniform relative or output-ULP target would imply a stronger
cancellation treatment than this initial algorithm promises. The negative
scale accounts for subtracting a reflection term from a quantity of size
log(1+abs(x)); it does not change the input used by the oracle. Keep intended
criteria separate from observed scan maxima. Assume nearest rounding and
gradual underflow. Check difficult references again at 1024 bits.

## Regions and error control

Use [recurrence and reflection](https://dlmf.nist.gov/5.5):
`psi(x+1)=psi(x)+1/x` and `psi(x)=psi(1-x)-pi*cot(pi*x)`.
Reduce the reflection phase before multiplying by pi, using the already
validated native pi-scaled layer. Treat tiny arguments separately so a
subnormal trigonometric denominator does not corrupt a large reciprocal.

For positive arguments, shift upward to a precision-dependent threshold and
use [the Bernoulli asymptotic expansion](https://dlmf.nist.gov/5.11.E2).
For positive real arguments, the omitted tail is bounded by the first omitted
term ([DLMF 5.11(ii)](https://dlmf.nist.gov/5.11.ii)). Generate Bernoulli
coefficients from an exact rational recurrence, with provenance and tail
bounds recorded, and construct them at the requested native precision.
Accumulate recurrence corrections with compensation. Avoid forming x*x in
the large-x region and avoid repeatedly rounding x+1 during a long shift.

No upstream implementation or coefficient table is copied. Before accepting
the algorithm, scan positive/negative zero neighborhoods, pole neighbors,
full exponent ranges, half-integers, phase reduction at large negative
arguments and every region boundary. MPFR is an oracle only.

For `abs(x) <= epsilon`, evaluate `-1/x - euler_gamma<Real>` directly.
The first omitted term is `zeta(2)*x`, so its error relative to the dominant
reciprocal is O(x^2), well below epsilon here. This includes signed zero and
avoids early range loss when a trigonometric intermediate would be subnormal.

For other positive inputs use the following thresholds and asymptotic orders.
The table gives conservative upper bounds on the first omitted term, evaluated
from exact Bernoulli fractions; roundoff is assessed separately.

| Native significand bits | Shift to at least | Retained Bernoulli terms | Absolute tail bound | Tail / epsilon |
| --- | ---: | ---: | ---: | ---: |
| 24 | 8 | 4 | <2^-37 | <6e-5 |
| 53 | 16 | 8 | <2^-70 | <3e-6 |
| 64 | 16 | 10 | <2^-79 | <9e-6 |
| 113 | 32 | 16 | <2^-136 | <5e-8 |

`scripts/generate-digamma-coefficients.py` uses the exact rational recurrence
`B_n = -sum_{k=0}^{n-1} binomial(n+1,k)*B_k/(n+1)`, starting with B_0=1.
It prints B_(2k)/(2k) and the remainder bounds. Numerators and denominators
fit signed64 and are converted directly to the native type before division.
No rounded double coefficient is widened. The tail polynomial uses FMA.

Compute each shifted denominator from the original x and its integer offset.
Neumaier compensated accumulation retains the cancellation residuals when
subtracting the reciprocal corrections from the logarithm and tail. For large
x, square its reciprocal, not x itself. The common path performs at most
32 recurrence divisions and 16 polynomial terms; it has no unbounded loop,
allocation or shared mutable state.

For negative nonintegers compute `psi(1-x)-pi<Real>/math::tanpi(x)` through
the same typed pi layer used by generic math. This keeps phase reduction at
the original argument, including large negative values. Negative integers
are recognized before evaluating the reflection denominator.

## Measurements and validation

GCC 13/glibc 2.39, MPFR512 oracle at exactly represented native inputs.
The source-header scan includes ordinary grids, 128 neighbors on each side
of positive/negative zeros and poles, joins, native exponent grids, reciprocal
overflow boundaries and a fixed-seed random sample. Worst sampled inputs
were repeated at 1024 bits and agreed when rounded back to 512.

| Format | Inputs | Maximum scaled error, rounded up | Worst input (round-trip decimal) |
| --- | ---: | ---: | --- |
| float32 | 45,393 | 3.092 | -3.731958866119384765625 |
| float64 | 45,491 | 3.438 | -2.70103092783505172036484509590081870555878 |
| binary80 | 47,153 | 2.435 | -1.71134020618556701028804140074512929459161 |
| binary128 | 47,155 | 3.104 | -1.68041237113402061855670103092783514685014 |

There were no sampled range/classification failures. These are observed
maxima in the explicitly defined epsilon-scaled measure, not output ULPs or
proven global bounds. Near a zero, this contract gives absolute accuracy and
does not guarantee relative digits or the sign of an arbitrarily tiny result.

The shared probes additionally use independent `psi(n)=H_(n-1)-gamma` and
half-integer harmonic identities, plus a precision-improvement case at 1/2.
All 30 targeted tests passed, covering native contracts, the shared native
and MPFR cases, existing scalar facade behavior and MPFR special-function
exact/unset/explicit-precision regressions. Eight native core tests
and the explanatory example also passed with MPFR disabled. Full integration
builds and CTest matrices are pending; targeted checks use compatible B1
libraries and do not replace that gate.

A small GCC13 -O2 timing experiment varied inputs near 1/2, -3.75 and 128,
25,000 calls per region. Observed microseconds/call were 0.012--0.031 for
float32, 0.020--0.043 for float64, 2.40--2.45 for binary80 and 10.1--13.2 for
binary128 on this host. A concurrent two-job build was running; these are
indicative costs, not portable performance guarantees. No existing native
digamma provider is replaced.

Run `digamma_example` for the precision ladder, negative noninteger values,
opposite signs across a pole and the finite result at the largest double.
