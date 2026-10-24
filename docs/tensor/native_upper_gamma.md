# Native upper incomplete Gamma

`math::upper_gamma(a,x)` evaluates the **unregularized upper** incomplete
Gamma at matching native float32/64/80/128 precision. Integer, mixed-native,
complex and conversion-only arguments are not accepted; convert explicitly.
Native calls do not take a Precision argument. The MPFR overload retains its
own exact/approximate and explicit-precision contracts. Enabling MPFR never
changes native evaluation into an MPFR call.

```cpp
auto tail = uni20::math::upper_gamma(-0.5, 1.0); // positive, about 0.17815
```

## Domain and numerical acceptance

For finite real a and x>0, the value is the positive integral
`F = integral_x^infinity t^(a-1)*exp(-t) dt`. Negative shape is supported;
there is no pole at a nonpositive integer when x>0. This API does not divide
by Gamma(a) to produce a regularized value.

The exceptional conventions match the existing MPFR interface (checked with
MPFR 4.2.1), including its analytic endpoint convention:

- NaN in either argument gives NaN.
- For finite a, x<0 gives NaN, x=+infinity gives positive zero, and either
  signed x=0 gives Gamma(a). At this endpoint, a negative integer gives NaN
  and a signed zero shape gives the corresponding signed infinity.
- Shape +infinity with finite x gives +infinity, including finite negative x;
  either infinite x gives NaN.
- Shape -infinity gives +infinity for 0<x<1, positive zero for x>=1,
  and NaN for x<=0.

For finite a and x>0, let eps be spacing at one and F the high-precision
reference at **exactly the represented arguments**. Set
`S = max(1, abs(log(F)), abs(a*log(x)), x)`. The intended absolute error bound,
established before implementation, is `64*eps*S*F + denorm_min/2`.
The exponent scale accounts for errors in constructing logarithms before
exponentiation, including cancellation in `a*log(x)-x`. It is not permission
to perturb inputs. For eps*S near one this target promises little relative
accuracy; classification and termination are tested separately.

| Region / all four native formats | Measure | Intended criterion | Guarantee level | Evidence |
| --- | --- | --- | --- | --- |
| Ordinary shapes, x>0, algorithm joins | Absolute error / bound above | <=1 | Empirical target | Mixed and deterministic mantissa grids pass |
| Shape near zero or negative integers, small x | Same measure at original arguments | <=1; no shape pole | Empirical target and domain contract | Neighbor, small-x and recurrence tests |
| x near positive shape; large-shape tails | Same measure | <=1; avoid overflowing intermediate complete Gamma | Empirical target and range contract | Shape-ratio and cancellation sweeps |
| Subnormal, zero, overflow neighborhoods | Finite bound and classification separately | No premature intermediate range loss | Empirical target and range contract | Adjacent-value sweeps pass |
| Exceptional values, x=0 | Classification, signs, endpoint Gamma | Conventions above | API contract | Native tests and MPFR contract checks |
| Shape one, integer and half-integer shape | Independent exp/erfc/Ei/finite sums | Accuracy at each precision and precision gains | Numerical evidence | Shared precision probes |
| Extreme finite parameters | Integral bounds or independently conditioned reference | Bounded work; no silent unconverged result | Algorithm requirement | Huge-parameter and classification probes |

This is not a universal correct-rounding or uniform-ULP guarantee. Evaluation
assumes nearest rounding and gradual underflow. Iterative paths have caps
proportional to the significand bit count; exhaustion returns NaN instead of
an unconverged approximation. No nonconvergence remains in the recorded scans.
There is no allocation, global precision, signgam access or hidden MPFR
execution in the native implementation. At x=0 the native tgamma provider
supplies the endpoint.

## Algorithm

The integral and recurrence are from [DLMF 8.2](https://dlmf.nist.gov/8.2) and
[8.8](https://dlmf.nist.gov/8.8). The implementation combines the
[series in 8.7](https://dlmf.nist.gov/8.7) and
[continued fraction in 8.9](https://dlmf.nist.gov/8.9):

1. For a>=1/2 and x<=a, sum the positive lower series and form its complement,
   where the upper regularized value Q stays away from zero. Combine log(Q)
   with log(Gamma(a)) before exponentiation.
2. In the upper tail, evaluate `U(a,x)=exp(x)*x^(-a)*Gamma(a,x)` by a fraction.
   For a<=0,x>=1, neighboring positive Stieltjes truncations supply the
   convergence test. Rounded endpoints are not interval enclosures.
   Elsewhere use the contracted fraction with denominators x+1-a,x+3-a,...
   and numerators k*(a-k).
3. For x<=1 and modest negative/small shape, reduce shape to [-1/2,1/2],
   evaluate a cancellation-safe series there, then recur down in scaled U.
   Large negative shapes use the fraction, avoiding work proportional to |a|.

The small-shape series uses `L(a)=log(Gamma(1+a))/a`, L(0)=-gamma:

```
U(a,x) = exp(x) * [expm1(a*(L(a)-log(x)))/a
                  - sum_{k>=1} (-x)^k/(k!*(a+k))].
```

Compute the first quotient through expm1(t)/t and its limit one. Evaluate L
by a shifted Stirling divided difference, retaining a rather than first
rounding 1+a. Shifts 16/32/32/64 and 4/8/10/16 terms give derivative-remainder
bounds below 2^-46.586/2^-87.980/2^-101.363/2^-170.058, by differentiating
the remainder in the positive-real [Binet integral](https://dlmf.nist.gov/5.9).
The exact-rational generator is `scripts/generate-upper-gamma-bounds.py`.
A 1024-point quotient grid has absolute errors at most
2.196/2.499/2.116/3.384 epsilons against MPFR1024. Truncation bounds alone
are not total floating-point error bounds.

At x=0,a=-b with positive integer b, the depth-n contracted fraction has
relative truncation error `1/binomial(b+n+1,b)`. Exact rational finite-fraction
checks agree. With b=8/16/16/32, depths 64/128/128/256 bound the error by
2^-33/2^-69/2^-69/2^-141. This explains the endpoint cutoff and iteration
scale; a positive-bracket-only iteration was ineffective at tiny x.

For a>2*max_exponent and x<=2a, integrating over [z,z+1], z=max(a,x), gives
the overflowing lower bound `a^(a-1)*exp(-2a-1)`. This avoids an expensive
transition iteration for a result known to overflow. The unregularized
function therefore does not need a general large-shape uniform transition
expansion merely to settle this region.

## Guarded range and cancellation evaluation

Ordinary paths use native pow/exp with compensated logarithmic scales.
Near a final overflow or subnormal boundary, or when
`eps*abs(a)*abs(log(x)) + eps*x > 1/32`, recompute the scaled tail and
logarithmic scale using two native words. A corresponding lower-complement
guard recomputes log Gamma, P and Q near that path's overflow boundary.
Guarding only the final exponent is insufficient: the scaled tail or complement
can itself contain the deciding error.

The private two-word helper uses TwoSum and FMA residuals, with division
refined by residual corrections. It supplies 40/85/96/145 working bits for
24/53/64/113-bit native inputs, leaving room below the two-word capacity for
accumulated roundoff. Log uses binary reduction and an atanh series; exp uses
log(2) reduction and a Taylor series. Subnormal exp results are rounded in
integer units of the smallest subnormal, retaining the low word through the
halfway decision. Exponent guards precede narrowing to a native int.

The tail's logarithmic scale is divided by a common power of two before
forming the two-word product and difference. The reduced logarithm has
magnitude below one, leaving native exponent range for both terms and their
residuals. Range comparisons use the same scale; only a bounded result is
rescaled. A finite fused estimate alone is insufficient: its intermediate
unfused product can overflow and poison the residual with NaN. The regression
`a=0x1.9d4d4ap+122f`, `x=0x1.cf67fep+126f` has this behavior; the integral over
`[x,x+1]` already proves that its result must overflow to positive infinity.
Corresponding cases are tested at every configured native precision.

The log(2) and log(2*pi) prefixes are reconstructed from exact 16-bit chunks.
The generator uses rational series bounds for log(2) and Machin pi, plus
160/240-digit Decimal logarithms for log(2*pi); the prefixes also agree with
an MPFR768 check. Guarded log Gamma uses shifts 32/64/64/128 with
4/8/10/16 Stirling terms, whose omitted-term bounds are
2^-55.214/2^-104.477/2^-122.256/2^-202.488.

For extreme cancellation, construct `(a/2)*log(x)-x/2` before doubling to
avoid overflowing the product itself. For a=-b and tiny x, integral bounds
permit U=1/b when its relative error is below guard resolution: use
x/(b-1) for b>1, or the conservative 2*sqrt(x) bound for b>=1/2.
These are bounded approximations, not clamps on an already rounded answer.

For small x and b>1, the guard can also use the integral
`U(-b,x) = integral_0^infinity exp(-x*t)/(1+t)^(b+1) dt`.
Integrating the finite Taylor expansion of exp gives terms
`t_n = (-x)^n/[b*(b-1)*...*(b-n)]` for n<b. The integrated Taylor
remainder after terms 0 through n-1 is bounded by `abs(t_n)`, since
`abs(R_n(x*t)) <= (x*t)^n/n!` and the corresponding moment exists.
When that bound is below the guarded tolerance, return the partial sum;
otherwise retain the continued-fraction path. This avoids slow fraction
convergence at modest negative shapes near small-x overflow boundaries,
without increasing the iteration cap or relaxing the error target.

## Evidence and limitations

The following GCC13 native scans call the actual `uni20::math::upper_gamma`
source with Uni20 MPFR execution disabled. Raw MPFR is used only by the
independent oracle. The numbers are **observed maxima**, not new guarantees.

| Grid | Inputs per format | float32 | float64 | binary80 | binary128 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Mixed shapes, endpoints and joins | 3192 | 0.935853 | 0.096364 | 0.149261 | 0.202375 |
| Deterministic mantissas, small-x and join neighbors | 4542 | 0.273603 | 0.096364 | 0.190929 | 0.248270 |
| Adjacent values around final range boundaries | 1664/1536/1536/1536 | 0.999207 | <1 | <=1 | <=1 |
| Varying shape, x/a from .75 to 1.25 near overflow | 528 | 0.00003338 | 0.00000460 | 0.000000304 | 0.000000307 |
| Modest negative shapes, small endpoints and range neighbors | 8721 | 0.040873 | 0.024624 | 0.046396 | 0.059256 |
| Larger negative shapes, independently located overflow boundaries | 3271 | 0.001466 | 0.002092 | 0.001351 | 0.000953 |
| Large-parameter cancellation | 448 | 0.006809 | 0.007458 | 0.008920 | 0.007429 |

Entries are fractions of the intended bound. All grids pass classification
and nonconvergence checks. Near a subnormal midpoint, the half-subnormal
allowance naturally dominates, giving a ratio close to one.

The modest-negative-shape scan uses b=-a on the grid 9/8,10/8,...,8,
plus each integer 1 through 8 and its immediate native neighbors. It samples
32 neighbors on either side of an approximate overflow endpoint, multiplicative
neighbors, powers of two, and normal/subnormal endpoint extremes. Its largest
errors occur at x=1/2: a=next_down(-1), next_down(-2), -5/4 and -11/8 for
float32/64/80/128 respectively.

The larger-negative-shape scan uses b=16,32,64,100,128,256,512,1000,2000,2048
and each value's immediate neighbors. MPFR512 independently locates the first
native endpoint whose result rounds finite; 32 endpoints in each direction
exercise the overflow transition. Additional endpoints multiply that boundary
by exp(j/(4*b)), j=-16,...,16, or use powers of two and native endpoint extremes.
The largest errors in that scan occur at x=1/2 and a=next_up(-16) for float32
and binary80, or next_down(-16) for float64 and binary128. The worst references
in both scans agree when repeated at 1024 bits. These wider negative shapes
exercise the bounded expansion at extended precision as well as float32.

The guard is covered by regressions for failures in earlier candidates:
float64 a=-100,x approximately .00078967680221933849 prematurely overflowed;
binary80 a=2000,x approximately 6051.189867266821 also overflowed early.
Varying positive shape exposed a separate lower-complement failure near
a=1755.5483429045. A huge-shape binary128 case a=2^113 lost the cancellation
in a*log(x)-x by many orders of magnitude. The guarded paths resolve these
cases without changing the acceptance bound. A further regression uses
float32 a=-3.5 and the represented x=1e-11: the old guarded fraction
exhausted its cap and returned NaN, while the bounded expansion yields the
finite correctly rounded value `0x1.0fe3b8p+126`. Shared probes cover this
point and range-boundary neighbors at modest negative shapes at every
configured native precision.

Ordinary references use MPFR512. Boundary identities use finite sums for
positive integers, erfc for half-integers, and Ei followed by recurrence for
nonpositive integers, adding oracle precision for recurrence cancellation.
The large-parameter falling-factorial reference uses its integral remainder
bound and adds argument-exponent bits to the oracle working precision. Worst
references agree on rounding at 1024 bits, except one large-parameter
binary128 reference differs by about 7.18e-155 relative between 512 and 1024
bits; an independent integer finite sum differs by about 2.07e-155. Both are
negligible against the native target.

Direct MPFR gamma_inc can be very slow for subnormal shape near zero. For
|a|<2^-256 and .49<x<3, selected probes use Gamma(0,x), adding a conservative
10*|a| derivative bound to the measured error. These are bounded limit
references, not claims of MPFR evaluation at the original subnormal shape.

The permanent tests cover native availability/domain rules, shared precision
identities, precision gains and independent range-boundary references.
The targeted upper-Gamma selection passes 22 MPFR-enabled tests and
14 native-only tests under GCC13, and 18 tests under Clang19 with binary128
disabled. Missing-provider executables and the example also pass. Removing the guarded recomputation in an isolated mutation makes
the accuracy probe fail at all four native precisions while both MPFR
cases still pass.

The preceding small-endpoint checkpoint passed the following full unfiltered
builds and CTest runs with examples, CLI and Python bindings enabled. The
subsequent scaled-product correction requires a fresh complete matrix before
publication:

| Configuration | CTest passes | Numerical tests | Runtime skips |
| --- | ---: | ---: | --- |
| GCC13 Debug, MPFR/MPC/MPLAPACK, native binary80/128 | 3358 | 635 | 1 NUMA topology test |
| GCC13 Release without MPFR, native binary80/128 | 2537 | 266 | 1 NUMA topology test |
| Clang19 Debug, MPFR/MPC/MPLAPACK, binary80; binary128 disabled | 3005 | 530 | 1 NUMA topology test |

The separate unfiltered numerical runs and their complete coverage reports
passed in all three configurations. Runtime skips are reported separately
from passes; there were no skipped numerical probes. Clang binary128 and
macOS were not exercised. CUDA and complex upper Gamma are outside this
implementation.

### Cost sample

An optimized GCC13 run on the ryzen host measured 1024 varying calls per
region, with no provider comparison because Uni20 had no native upper-Gamma
implementation. Small-shape / ordinary-tail calls took about 1.04/.34 us
for float32, .36/.20 us for float64, 2.91/.84 us for binary80, and
79.7/20.9 us for software binary128. A sampled guarded subnormal call took
about 4.1 us for float32 and 4.4 us for float64. These are local timings,
not performance guarantees; guarded work is deliberately more expensive
than the ordinary path. No existing provider was replaced.
