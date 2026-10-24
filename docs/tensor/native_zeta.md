# Native Riemann zeta investigation

The immediate gap is binary128 `math::zeta(x)`. Investigate a reusable native
implementation without Boost, MPFR evaluation or global state. Existing
float/double/long-double providers remain selected until an accuracy and
performance comparison justifies changing that choice. This document records
acceptance before prototyping; no new public overload is implemented yet.

## Domain and acceptance

Use the real continuation, with a pole at one, exact zeros at negative even
integers, zeta(0)=-1/2 and limit one at positive infinity. Match the verified
MPFR conventions: NaN and negative infinity give NaN, the pole gives positive
infinity, and exact negative even integers give positive zero. No complex
function or additional precision argument is added.

For f=zeta(x) at the exact represented input, let eps be spacing at one.
The initial empirical target is `32*eps*abs(f)` for nonnegative arguments.
For negative nonzero results, use
`32*eps*max(1,abs(log(A)))*abs(f) + denorm_min/2`, where
`A=2*(2*pi)^(x-1)*Gamma(1-x)*zeta(1-x)` is the positive reflection amplitude.
Its logarithmic factor accounts explicitly for native log-domain evaluation,
not uncertainty in the input. It is not a universal ULP or correct-rounding
guarantee. Exact input reduction should preserve relative accuracy close to
trivial zeros; zero values themselves require exact classification instead.

| Region / all four native formats | Measure | Intended criterion | Status |
| --- | --- | --- | --- |
| Nonnegative finite x, excluding one | Original-input absolute error | <=32*eps*abs(f) | Unmeasured |
| Neighbors of the pole | Relative error at the represented input | <=32*eps; preserve sided sign | Unmeasured |
| Negative ordinary x and neighbors of trivial zeros | Reflection-scaled relative error plus final subnormal allowance | Bound above; preserve nonzero sign unless underflow | Unmeasured |
| Negative even integers | Exact value | Zero, without evaluating an overflowing amplitude | Unmeasured |
| Negative overflow neighborhoods | Classification and finite accuracy separately | No premature intermediate overflow | Unmeasured |
| Large positive x | Distance from limiting value | Return one only once the omitted tail is below half a spacing | Unmeasured |
| Independent integer values and precision ladder | Bernoulli/pi identities | Document accuracy improvements with precision | Unmeasured |

Worst cases will use MPFR512 references repeated at 1024 bits. Scans must
include adjacent native values around negative even integers, the pole and
range boundaries, plus ordinary grids and deterministic exponent samples.
Do not ask MPFR to evaluate enormous negative noninteger arguments when a
separate analytical range argument establishes native overflow.

## Candidate algorithms

For x>=1/2, use an Euler–Maclaurin tail after a finite Dirichlet sum. Handle
the pole explicitly and sum with compensation. For x>=p+1, p significand
bits, the integral bound on the omitted Dirichlet tail is below half a spacing
at one, so the rounded limit is one.

With terms through n=N-1 and M Bernoulli corrections, the absolute integral
remainder for positive s is at most
`abs(B_(2M)/(2M)!)*(s)_(2M-1)*N^(1-s-2M)`. On each unit interval [j,j+1],
bound the rising factorial at j+1 and the decreasing power at j. The generator
`scripts/generate-zeta-coefficients.py` uses exact rational arithmetic for
these bounds and prints 128-bit binary coefficient prefixes.

| Format | N | M | Absolute remainder / eps, upper bound |
| --- | ---: | ---: | ---: |
| float32 | 8 | 4 | 0.016667 |
| float64 | 16 | 8 | 0.001732 |
| binary80 | 16 | 10 | 0.004037 |
| binary128 | 32 | 16 | 0.000054 |

The bounds cover 0<s<=p+1 and control truncation only. Native coefficient,
power, summation and reflection errors need separate numerical validation.

For x<0, use reflection, preserving sin(pi*x/2) through the existing pi-scaled
reduction. Detect exact even integers before Gamma or exponential evaluation.
The reflection amplitude must be evaluated in scaled/log form, with its sine
factor included before deciding final range. Accumulate the Stirling logarithm
using product residuals and compensated addition. Preserve the accumulated
low part in the final exponential scaling; rounding the whole logarithm first
can move a finite binary128 result past overflow.
Keep the sine factor multiplicative: taking its logarithm loses relative
accuracy close to trivial zeros. Below the Stirling threshold, the finite
product `Gamma(t)*(2*pi)^(-t)*(t/pi)*zeta(1+t)`, t=-x, also avoids cancellation
between large logarithms and rounding the Gamma argument from t to 1+t.

For abs(x)<1/2, compute `delta*zeta(1+delta)`, delta=-x, directly from the
Euler–Maclaurin formula. The singular tail becomes `N^(-delta)` and the
remaining terms are multiplied by delta. Combine it with
`sinpi(x/2)/delta` in reflection. This avoids the severe near-zero cancellation
of direct Euler–Maclaurin evaluation, and rounding 1-x to one cannot introduce
a spurious pole. For abs(x)<eps/2 the first-order expansion
`-1/2-x*log(2*pi)/2` suffices; its omitted term is O(eps^2).

## Current investigation evidence

The local prototype's 1513-input near-zero/pole scan per format passed the
target at MPFR512, with worst references agreeing at 1024 bits. Largest
fractions of the bound were .129/.150/.238/.342 for float32/64/80/128.
These are observed sample maxima, not global guarantees.

Independently bisected overflow boundaries in the first several overflowing
negative lobes were checked at 1536 neighboring native inputs per format.
The compensated logarithm and final scaling passed classification checks for
all four formats. A regression must retain the binary128 argument
`-2312.9484675583841235265933212349314204917536313162`, where rounding the
logarithm before exponentiation gave infinity for a finite result.

A separate 384-input scan of neighbors of -8, -16 and -32 passed after
preserving the multiplicative sine and finite Gamma product. Largest bound
fractions were .484/.671/.562/.110; worst references agree at 1024 bits.

Broad scans, provider comparisons, permanent tests and public integration are
still pending. Prototype evidence is not full implementation validation.

Sources: [DLMF reflection formulas](https://dlmf.nist.gov/25.4),
[Euler–Maclaurin representations](https://dlmf.nist.gov/25.11#iii), and
[integer values](https://dlmf.nist.gov/25.6). These are mathematical identities;
no implementation or coefficient table is copied from another library.
