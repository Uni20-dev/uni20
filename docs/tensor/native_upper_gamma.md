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
