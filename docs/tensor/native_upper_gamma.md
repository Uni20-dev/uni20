# Native upper incomplete Gamma investigation

This is the design and numerical acceptance checkpoint for native
`math::upper_gamma(a,x)`, the **unregularized upper** incomplete Gamma.
The native implementation is not yet integrated. The existing MPFR interface
and its explicit-precision behavior remain unchanged. Native arguments must
stay native at every configured float32/64/80/128 precision, without Boost,
MPFR evaluation or mutable global state.

## Domain and acceptance established before prototyping

For finite real a and x>0, evaluate the positive integral
`F = integral_x^infinity t^(a-1)*exp(-t) dt`. Negative shape is supported;
there is no pole at a nonpositive integer when x>0.

Match the existing MPFR conventions at exceptional inputs, including its
analytic endpoint convention rather than replacing it with a divergent-integral
interpretation:

- NaN in either argument gives NaN.
- For finite a, x<0 gives NaN, x=+infinity gives positive zero, and either
  signed x=0 gives Gamma(a). At the latter endpoint, a negative integer gives
  NaN and a signed zero shape gives the corresponding signed infinity.
- Shape +infinity with finite x gives +infinity, including finite negative x;
  either infinite x gives NaN.
- Shape -infinity gives +infinity for 0<x<1, positive zero for x>=1,
  and NaN for x<=0.

These conventions were checked directly with MPFR 4.2.1; they do not establish
native implementation evidence.

Let eps be spacing at one and F the high-precision reference evaluated at
exactly the represented arguments. For finite a and x>0, define
`S = max(1, abs(log(F)), abs(a*log(x)), x)`. The intended absolute error bound is
`64*eps*S*F + denorm_min/2`. Evaluate this measure in a wider reference type.
The explicit exponent scale accounts for errors in constructing a logarithm
before exponentiation, including cancellation between `a*log(x)` and x. It
is not an allowance to perturb the input. It is not a uniform ULP or
correct-rounding guarantee, and for parameters so large that eps*S approaches
one it promises little relative accuracy. Range behavior and iteration
termination still need independent validation there.

| Region / all four native formats | Measure | Intended criterion | Guarantee level | Evidence |
| --- | --- | --- | --- | --- |
| Ordinary positive/negative shape, x>0 and algorithm joins | Absolute error divided by the bound above | <=1 | Empirical target | Not measured |
| Shape near zero or negative integers, small x | Same measure at original inputs | <=1; no spurious shape pole | Empirical target and domain contract | Not measured |
| x near positive shape, and large-shape tails | Same measure | <=1; scaling must not form overflowing Gamma(a) for a finite result | Empirical target and range contract | Not measured |
| Subnormal, zero and overflow neighborhoods | Finite bound and classification separately | No premature intermediate overflow/underflow | Empirical target and range contract | Not measured |
| Exceptional arguments and x=0 | Classification, signs and endpoint Gamma comparison | Conventions above | API contract | MPFR contract probe only |
| Shape one, half-integers and integer recurrence | Independent exp/erfc/finite sums | Accuracy at each precision; demonstrate improved accuracy with precision | Identity and numerical evidence | Not measured |
| Extreme finite parameters | Range bounds or independently conditioned reference | Bounded work; no silent unconverged approximation | Algorithm requirement | Not measured |

Use MPFR512 initially, repeating worst cases at 1024 bits and increasing
precision further when cancellation in an independent reference requires it.
The MPFR manual warns that gamma_inc itself can overflow internally or be slow
for large shapes/results; distinguish oracle failure from native failure.

## Candidate decomposition

The defining integral and recurrence are from
[DLMF 8.2](https://dlmf.nist.gov/8.2) and
[8.8](https://dlmf.nist.gov/8.8). Series are from
[8.7](https://dlmf.nist.gov/8.7), continued fractions from
[8.9](https://dlmf.nist.gov/8.9), and large-argument behavior from
[8.11](https://dlmf.nist.gov/8.11). No implementation or coefficient table is
copied from a provider.

Investigate three complementary paths:

1. For a>=1/2 and x<=a, a positive lower series gives P(a,x); use its
   complement only where Q is bounded away from zero. Multiply the complete
   Gamma scale by Q without first constructing an overflowing Gamma(a).
   Very large shapes in this region have an overflowing unregularized result;
   establish an integral lower bound before using that shortcut.
2. In the upper tail, evaluate the scaled quantity
   `U(a,x)=exp(x)*x^(-a)*Gamma(a,x)` by continued fractions. For a<=0 the
   positive Stieltjes coefficients provide neighboring exact-arithmetic
   truncations that bracket the result. Rounded truncations are not rigorous
   interval enclosures. For positive shape above the complement region,
   investigate the contracted fraction and its convergence separately.
3. For x<=1 and modest negative/small shape, reduce shape to [-1/2,1/2],
   evaluate a cancellation-safe series there, and recur down in the scaled U.
   Avoid dividing a subtraction of independently rounded complete/lower Gamma
   values by a near-zero shape. Large negative shapes should use a bounded
   continued fraction rather than a recurrence proportional to abs(a).

For the small-shape series, write `L(a)=log(Gamma(1+a))/a`, with L(0)=-gamma.
Then

```
U(a,x) = exp(x) * [expm1(a*(L(a)-log(x)))/a
                  - sum_{k>=1} (-x)^k/(k!*(a+k))].
```

Form the first quotient using `expm1(t)/t` and its limit one. Obtain L from a
shifted Stirling difference and `log1p` divided differences, using existing
exact Bernoulli coefficients. This is a candidate to test, not yet an accepted
accuracy claim. A precision-dependent shift must bound the differentiated
Stirling remainder and preserve small a without first rounding 1+a.

Accumulate logarithmic scales with FMA residuals and compensated addition.
Preserve their low part during final exponentiation. A mathematically
representable final value must not be lost because the separate power,
complete Gamma or exponential overflows. Determine finite iteration caps,
convergence checks and failure behavior before integrating the public adapter.

The positive-shape uniform transition formulas in
[DLMF 8.12](https://dlmf.nist.gov/8.12) remain a possible alternative if
ordinary fractions/series prove inadequate. Do not implement them merely
because regularized Gamma libraries need them: the unregularized function
overflows much earlier near large positive shape.

## Next numerical checkpoint

Prototype small shape with tiny x, negative-integer neighbors, x near a,
large positive shape with x far beyond a and a finite result, and final range
boundaries. Keep observed maxima separate from the intended table. Establish
whether this decomposition is workable before adding a public native overload.

## Initial prototype evidence

The first positive Stieltjes truncation pair did not converge for tiny x,
including a=-8 and x=2^-48 in float32. Contracting to the fraction with
successive denominators `x+1-a, x+3-a, ...` and numerators `k*(a-k)`
removes that ineffective endpoint enclosure. Use it for large negative shape
at small x; retain the positive bracketing form where x>=1. At x=0, a=-b
with positive integer b, the depth-n contracted fraction has exact relative
truncation error `1/binomial(b+n+1,b)`. Exact rational checks of the finite
fraction agree with this identity. With b=8/16/16/32, depths 64/128/128/256
already give errors below 2^-33/2^-69/2^-69/2^-141. These endpoint bounds
justify the scale of the iteration cap, not a total floating-point bound.

For the small-shape log-Gamma quotient use shifts 16/32/32/64 and
4/8/10/16 Stirling terms. The divided-difference remainder is bounded by the
first omitted term of the differentiated Stirling series at `shift-1/2`.
This follows by differentiating the remainder in the positive-real
[Binet integral](https://dlmf.nist.gov/5.9). The exact-rational generator
`scripts/generate-upper-gamma-bounds.py` gives bounds below
2^-46.586/2^-87.980/2^-101.363/2^-170.058. This bounds truncation only.
A 1024-point grid in [-1/2,1/2], excluding zero, gives maximum absolute
quotient errors of 2.196/2.499/2.116/3.384 epsilons against MPFR1024.

The prototype's initial mixed grid has 3192 input pairs per format: half-unit
shape steps, eight ordinary endpoints, negative-integer/zero/join neighbors,
small powers of two, deterministic native mantissas and large positive shapes.

| Format | Largest fraction of intended bound | Worst represented shape | Worst represented endpoint |
| --- | ---: | --- | --- |
| float32 | 0.935853 | -11.33757877349853515625 | 52.50345611572265625 |
| float64 | 0.096364 | -0.499999999999999722444243843710864894092082977 | 0.500000000000000555111512312578270211815834045 |
| binary80 | 0.142906 | -0.50000000000000000037947076036992655190260848 | 0.99999999999999999962052923963007344809739152 |
| binary128 | 0.196020 | -0.500000000000000000000000000000000481482486097 | 0.999999999999999999999999999999999518517513903 |

No accuracy, nonconvergence or range-classification failures occurred in this
revised grid. Worst references agreed at 1024 bits. The largest float32
fraction occurs in the subnormal tail. These are prototype observations,
not a full native implementation or proof of a domain-wide guarantee.

MPFR gamma_inc is very slow for subnormal shapes near zero: an isolated
binary80 smallest-subnormal shape with x=1/2 did not finish within five seconds.
For the 42 such samples at each of float64/binary80/binary128, use Gamma(0,x)
and bound the reference uncertainty by `10*abs(a)` for abs(a)<2^-256 and
0.49<x<3. Differentiating the defining integral and bounding its pieces below
and above one gives this conservative derivative bound. Add the uncertainty
to the measured error. This is an independently bounded limit reference,
not a claim that MPFR evaluated the original subnormal shape. Other samples
use MPFR512 at the original inputs.

Two additional range defects were reproduced and corrected in the candidate:

- With a=2 or 10000 and endpoint near the native maximum, exponent splitting
  could form zero times an infinite low-part exponential. Renormalize the
  compensated logarithm before splitting and handle a zero/infinite half
  exponential before multiplying its correction.
- At a=2^p and x=nextafter(a,+infinity), all four formats exhausted the
  continued-fraction cap even though the integral certainly overflows. For
  a>2*max_exponent and x<=2a, integrate over [z,z+1], z=max(a,x), to obtain
  the lower bound `a^(a-1)*exp(-2a-1)`. It exceeds every supported format's
  maximum. Return infinity from this bound before iteration. The regression
  now returns infinity at each precision. This argument also bounds work
  around very large shapes without requiring a uniform transition expansion.

The API remains unimplemented. Wider mantissa/exponent sampling, independently
located final range boundaries, large-parameter cancellation, performance,
permanent tests and full configuration validation remain required.

### Final-range investigation remains unresolved

The neighboring-value boundary sweep finds classification errors even though
all finite comparisons meet the empirical error bound. With direct half-powers
for nonpositive shape, the 1664/1536/1536/1536-pair sweep reports 6/6/9/3
mismatches for float32/64/80/128, including repeated neighboring inputs.
For example, float64 a=-100 and
x=0.00078967680221933849067650346142954731476 gives premature infinity;
binary80 a=2000 and x=6051.1898672668210741321104251255746931 also gives
premature infinity. Positive large-shape exponent cancellation and final
rounding of the scaled fraction both need more guard precision. Direct
half-powers improve some cases but do not settle the range contract.

The boundary oracle now uses independent exact identities: positive integer
shapes use their finite falling-factorial sum, shape1/2 uses erfc, nonpositive
integers use E1 followed by recurrence, and shape-1/2 uses the half-integer
recurrence. Recurrence references gain guard bits proportional to the number
of steps and log2(x) to compensate for cancellation. All 36 ordinary cross-checks
agree with MPFR gamma_inc at 512 bits; worst boundary references agree at 1024.
This avoids very slow direct incomplete-Gamma evaluation in extreme tails.

The next bounded experiment is a native guarded evaluation near final range
boundaries, retaining extra words in the fraction and exponent rather than
clamping a rounded answer. Existing FMA-based expansion work is relevant, but
native execution must still remain independent of MPFR. Keep these failures
visible until corrected; do not integrate or publish the native overload yet.
