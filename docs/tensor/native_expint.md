# Native real exponential integral

`math::expint(x)` evaluates the real principal-value Ei at every configured
native precision, including binary128. Native evaluation uses Uni20 algorithms
without Boost or MPFR. The former standard-library path is replaced after the
accuracy and performance comparison below. MPFR scalar semantics are unchanged.

## Acceptance established before prototyping

For native evaluation, NaN gives NaN, either zero gives negative
infinity, positive infinity gives positive infinity, and negative infinity
gives negative zero. Finite negative inputs are negative, tending to negative
zero through gradual underflow. Finite positive results may overflow only
at the final native range boundary. The MPFR overload keeps its own
provider and explicit-precision behavior.

Let eps be spacing at one and f=Ei(x) evaluated at the exact represented
input. Use scale=max(1,abs(f)) when 0<x<1, and abs(f) elsewhere. The intended
absolute error target is `32*eps*scale + denorm_min/2`. This gives an absolute
criterion near the positive zero and a relative criterion on the negative
tail, with an allowance for final subnormal rounding. It is an empirical
acceptance target, not a correctly-rounded guarantee or input-conditioning
allowance. Compare to MPFR512 and repeat hard cases at 1024 bits.

| Region / all native formats | Measure | Intended criterion | Guarantee level | Observations |
| --- | --- | --- | --- | --- |
| Finite ordinary arguments and joins | Absolute error divided by the bound above | <=1 | Empirical target | Wide scan and permanent tests pass |
| Positive zero neighborhood | Absolute error, scale one | <=32*eps | Empirical target | Wide scan and permanent tests pass |
| Negative tail and subnormal/zero boundary | Relative bound plus final half-subnormal allowance | <=1; no premature loss from intermediates | Empirical accuracy and range contract | Wide scan and permanent tests pass |
| Positive overflow boundary | Finite bound and classification checked separately | No premature intermediate overflow | Empirical accuracy and range contract | Wide scan and permanent tests pass |
| Zeros, infinities and NaN | Classification and zero sign | Stated contract | API contract | Wide scan and permanent tests pass |
| Independent small-argument series and precision ladder | Wider defining-series reference | At least sixteen-fold accuracy gain on controlled cases | Demonstrated examples | Wide scan and permanent tests pass |

## Algorithms and error control

Use [DLMF 6.6](https://dlmf.nist.gov/6.6) for the power series,
[6.9](https://dlmf.nist.gov/6.9) for the positive E1 continued fraction, and
[6.12](https://dlmf.nist.gov/6.12) for asymptotics and remainder bounds.
No provider implementation or coefficient table is copied. Let p be the
number of significand bits.

For -1<=x<2p, accumulate
`Ei(x)=gamma+log(abs(x))+sum(x^k/(k*k!))` with compensation. Limit the
negative series to this region to avoid severe cancellation. Once k>=2*abs(x),
the remaining terms decrease geometrically with ratio below 1/2. Stop when
the current term is at most eps*max(1,abs(sum))/16. The absolute scale also
protects cancellation near the positive zero. The cap is 8p+64 terms.
For positive x<2p, all intermediate terms fit the native exponent range.

For x<-1, evaluate Ei(x)=-E1(-x) with a positive continued fraction for
`exp(y)*E1(y)`, y=-x. Neighboring even/odd truncations bracket the exact value
in exact arithmetic. Evaluate both backwards without cancellation or products
of forward-update factors, doubling depth from 16 until the computed gap is
at most 4*eps times the value. The depth cap is 16p (largest power of two no
greater than that cap). Rounded endpoints are not rigorous interval enclosures;
this stopping test and the accuracy target are verified empirically, not
claimed as a formal total-roundoff bound.

For x>=2p, sum the factorial asymptotic series for `x*exp(-x)*Ei(x)` with
compensation. After adding t_k=k!/x^k, stop when
`t_k<=eps*sum/(4*(k+2))`. Terms are still decreasing since k<=p. The DLMF
positive-Ei remainder factor is bounded by k+4, so omitted terms contribute
less than eps*sum/2 at this stopping point. The cap is p terms. This bound
controls truncation, not all floating-point error in the recurrence.

For both tails, evaluate `exp(x/2)*(exp(x/2)*factor)`. The inner product stays
normal near the final subnormal boundary and finite near the final overflow
boundary. Only the final multiply determines native range there. Conservative
input cutoffs far beyond the actual result boundaries avoid work on enormous
arguments with known infinity/negative-zero results. If an iteration cap is
exhausted, return NaN rather than a silently unfinished approximation; no such
failure occurred in the scans or permanent tests.

## Observed accuracy

References use MPFR512 at the original represented input, with worst cases
repeated at 1024 bits. The broad scan includes 32769 rational-grid arguments,
64 neighbors of joins and the independently bisected positive root, powers of
two and deterministic native-mantissa/exponent samples. Arguments whose
magnitude exceeds 24000 are covered by explicit range tests instead of asking
MPFR to evaluate exponentials of astronomically large arguments.

| Format | Broad inputs | Largest fraction of acceptance bound | Worst input |
| --- | ---: | ---: | --- |
| float32 | 38757 | 0.994348 | -93.81081390380859375 |
| float64 | 40001 | 0.452130 | -726.46108713093053665943443775177001953125 |
| binary80 | 70587 | 0.534267 | 127.99999999999999989591659144139157433528 |
| binary128 | 70611 | 0.635903 | 225.21621621621621621621621621621621554995 |

A separate 40384-input scan per format concentrates random mantissas in the
positive series region and negative continued-fraction region. Its maxima
were .267201/.399613/.534267/.681430 respectively; the binary128 worst input
was 224.09975081835361660764610062665121046283. Another 384 inputs per format
straddle independently located normal/subnormal, zero and overflow boundaries.
All passed, with no premature range failures. Boundary error fractions can
approach one because the additive allowance is half a subnormal spacing.
These are sample observations, not a proof over every input.

The shared registry covers all native types and MPFR128/256. Independent
references include a 400-term series at +/-1/4 and the upper-Gamma identity
Ei(-n)=-Gamma(0,n), including n=100. Precision gains are measured at 1/4.
Core regressions include finite Ei after exp(x) itself overflows, negative
subnormal tails and value classification. The example explains these cases.

## Provider comparison and selection

The same broad scan exposed substantial GCC13 libstdc++ errors. At x=-100,
its double/long-double implementation returns essentially -exp(-100)/100,
about 1% away from Ei(-100). Inspection of `tr1/exp_integral.tcc`, function
`__expint_E1_asymp`, explains this: its loop index is unsigned, so negating
that index produces a large positive integer; the first asymptotic correction
is rejected and only the leading term remains. The float provider also
overflows prematurely, for example at x=88.72972869873046875 where Ei is
approximately 3.90609272221689e36. These observations concern the tested GCC13
provider; they are not claims about every standard-library implementation.

Consequently Uni20 selects its native algorithm for all four formats. This
has a cost: representative 10000-call loops on the local GCC13 Linux host
measured roughly .019--.55 us for float32, .017--1.45 us for float64,
.061--1.71 us for binary80 and 2.6--128 us for binary128. Standard-provider
measurements were roughly .008--.062/.011--.414/.044--.678 us for the first
three types. In the positive series region the replacement can be tens of
times slower than an early standard-library asymptotic path. The negative
continued fraction near -1 dominates binary128 cost. These measurements were
taken during an unrelated build and are orientation, not timing guarantees.
Correctness and full-range behavior justify this documented tradeoff; native
execution still has no allocation or mutable shared state.

The targeted MPFR-enabled run passed 77 tests; the native-only run passed 12.
The synthetic missing-standard-provider executable compiled and ran both with
and without MPFR. This establishes fallback wiring, not second-platform
numerical validation. The explanatory example also compiled and ran.

Full GCC Debug, GCC Release without MPFR and Clang Debug integration remain
queued before publication. No macOS, CUDA or Clang binary128 validation is
claimed by this local checkpoint.
