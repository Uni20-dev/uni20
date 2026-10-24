# Native real Airy Ai

`math::airy_ai(x)` evaluates real Ai for configured float32/64/80/128 and
returns the input type. Native evaluation uses native arithmetic, a private
four-word expansion, and bounded integer phase reduction. It has no Boost,
MPFR or new runtime dependency. The MPFR overload retains its separate
working-precision rules and provider limitations.

```cpp
#include <uni20/core/math.hpp>
auto oscillatory = uni20::math::airy_ai(-10.0);
auto decaying = uni20::math::airy_ai(10.0L);
```

Integer inputs, conversion-only classes, complex values and native trailing
`Precision` arguments are not accepted. Convert explicitly to the desired real
type. Ai derivatives, Bi, complex Airy functions, tensors and CUDA are outside
this interface. The [example](../../examples/common/airy_example.cpp) explains
the two real regimes and demonstrates a huge negative argument.

## Domain and numerical acceptance

Ai is real and finite for every finite real argument. Both signed zeros give
the same positive Ai(0). NaN gives NaN; both infinities give positive zero.
These exceptional conventions match the existing MPFR API and were checked
with MPFR 4.2.1. Finite positive results may round to positive zero.

Evaluate references at the exact represented input. Let eps be spacing at one,
d the smallest positive subnormal, and F=Ai(x). For x<=0 use the positive
amplitude scale `A(x)=(1+abs(x))^(-1/4)/sqrt(pi)`. For x>0 let
`zeta=2*x^(3/2)/3`, evaluated in reference precision.

The following targets were set before prototyping; they are empirical
accuracy requirements, not proofs of correct rounding for every input.
Nearest rounding and gradual underflow are assumed.

| Domain / all four native formats | Error measure | Intended bound |
| --- | --- | --- |
| Finite x<=0, including zeros and region joins | Absolute error divided by A(x) | 64*eps |
| Finite x>0, central and asymptotic regions | Absolute error | 64*eps*max(1,zeta)*F + d/2 |
| Huge negative arguments | Same amplitude-scaled absolute error | 64*eps, without growth proportional to phase |
| Positive underflow boundary | Error bound above and classification separately | No premature zero from intermediate exponentials |
| Zero and nonfinite inputs | Analytic value, classification and sign | Conventions above |
| Increased precision | Error at selected nonzero values | Measured improvement over lower precision |

Relative error is inappropriate near negative zeros. The negative-axis bound
does not treat a lost phase as input uncertainty. The positive exponent scale
allows native rounding in constructing the decay exponent. A bounded iteration
which cannot meet its stopping criterion returns NaN; finite test inputs must
not encounter that path.

## Central expansion and asymptotic joins

The central expansion follows the two Maclaurin solutions of y''=x*y:
`Ai(x)=Ai(0)*F(x^3)+Ai'(0)*x*G(x^3)`. Their successive term ratios are
`x^3/(3k*(3k-1))` and `x^3/(3k*(3k+1))`; see
[DLMF 9.4](https://dlmf.nist.gov/9.4). Four nonoverlapping native words retain
the cancellation on the positive side. Rounded single-word initial constants
would lose too much accuracy there. Series stop at a relative term scale
`2^(-4p+8)` and have a `4p+64` cap, where p is significand bits.

Outside the central interval, the decaying and oscillatory expansions use
`u_0=1`, `u_k/u_(k-1)=(6k-5)*(6k-1)/(72k)` and inverse zeta computed as
`(1.5/abs(x))/sqrt(abs(x))`. This avoids an overflowing phase intermediate.
The real-variable remainder is bounded by the first omitted term in each
series; see [DLMF 9.7](https://dlmf.nist.gov/9.7), particularly 9.7.2, 9.7.5,
9.7.9 and section iii. Stop below eps/32 before terms grow, with the same
`4p+64` cap. These joins leave truncation room beneath the numerical target:

| Significand p | Central interval | Lower bound on zeta at join | Order reaching term <2^(-p-8) |
| --- | --- | --- | --- |
| 24 | [-8,8] | 15 | 10 |
| 53 | [-12,12] | 27 | 20 |
| 64 | [-14,14] | 34 | 23 |
| 113 | [-20,20] | 59 | 39 |

[`generate-airy-bounds.py`](../../scripts/generate-airy-bounds.py) checks these
bounds using exact rational arithmetic and integer square roots. Region
neighbors and both sides of the joins are covered by shared precision tests.

Positive tails normally use two half-exponentials to avoid premature
underflow. Within eight logarithmic units of the smallest subnormal, recompute
the exponent, series and logarithmic prefactor with four native words. Round
the result in integer subnormal units, retaining the low words through the
rounding decision. An initial single-word tail lost finite subnormals at
float32 x=28.5 and 28.5625 and exceeded the bound at x=28.375; this guard fixes
those failures. For sufficiently large positive x, the Ai upper bound in
DLMF 9.7.15 proves zero without constructing an overflowing exponent.

## Negative-axis phase reduction

For v=-x>0, write the exact binary input as `v=M*2^e`, with integer M.
Reduce the phase from M and e, rather than from a rounded native square root.
If 3e is odd, multiply M^3 by two before halving the exponent. This gives
`v^(3/2)=sqrt(N)*2^E` with integer N and E.

Set `P=p+32`, `K=max(0,E+P)`, and compute the exact integer
`S=floor(sqrt(N*2^(2K)))` by restoring binary square root. Read shifted
radicand bits without allocating the shifted integer. Multiply S by a prefix
of `4/(3*pi)`, retaining the low quadrant and fractional bits of
`4*v^(3/2)/(3*pi)`. The root truncation loses less than `2^-P` phase units;
the constant prefix contributes less than `2^(-P-15)`. Together they lose
less than `2^(-p-31)` before native conversion.

The final sine and cosine receive only an angle in [-pi/4,pi/4]. Quadrant
selection supplies sin(zeta+pi/4) and cos(zeta+pi/4). No enormous native phase
is formed. Fractional rounding across a quadrant boundary remains continuous
in the resulting sine/cosine pair.

The implementation uses private fixed-capacity uint32 limbs and uint64
products. Capacity follows the scalar's significand and maximum exponent;
there is no heap allocation or public arbitrary-integer API. The largest
supported format needs fewer than 25,000 root bits and 50,000 product bits.
Ordinary input exponents use a smaller stack capacity. Work is bounded but
grows quadratically with the required phase bits. The implementation explicitly
supports binary formats with at most 113 significand bits and maximum exponent
16384, covering the configured native types.

## Reproducible constants

[`generate-airy-phase.py`](../../scripts/generate-airy-phase.py) uses exact
integer Machin intervals for pi, then reciprocates the interval. Both endpoints
give the same 24,832-bit prefix of 4/(3*pi). It uses 64 guard bits and term
budgets 5359/1576 for atan(1/5) and atan(1/239); the conservative interval
uncertainty is 92,048 guard-scale units. An independent MPFR calculation at
25,088 bits reproduced the entire prefix.

[`generate-airy-constants.cpp`](../../scripts/generate-airy-constants.cpp)
generates 512-bit prefixes for Ai(0), abs(Ai'(0)), log(2) and log(pi)-1 at
2048-bit MPFR precision. It cross-checks Ai(0) against its Gamma identity and
the derivative against a central difference of MPFR Ai at +/-2^-400. The
identity agreed at 2048 bits; the derivative discrepancy was about 2^-804.
MPFR is an offline generator dependency only. Constant assembly retains
`4p+16` bits before four-word evaluation.

```bash
python3 scripts/generate-airy-phase.py --output /tmp/airy-phase.hpp
python3 scripts/generate-airy-bounds.py
c++ scripts/generate-airy-constants.cpp -lmpfr -lgmp -o /tmp/airy-constants
/tmp/airy-constants > /tmp/airy-initial-values.hpp
```

Generated coefficient words match the source headers; formatting is separate.
No approximation table was copied from another numerical provider.

## Numerical evidence and remaining validation

The actual CPO was checked in a native-only GCC13 configuration, with raw MPFR
used solely by the external oracle. Direct MPFR Ai is practical in a limited
range: its manual warns that arguments should typically have magnitude below
500. Outside that range, tests use an independent high-precision asymptotic
reference, explicit truncation control and enough phase bits for the original
input exponent. That reference agrees with direct 512-bit MPFR Ai at +/-129,
+/-160 and +/-256. Worst-case references were repeated at 1024 bits.
See the [MPFR Ai reference](https://www.mpfr.org/mpfr-current/mpfr.html#index-mpfr_005fai).

| Experiment | Samples per native format | Observed maximum fraction of intended error bound, 32/64/80/128 |
| --- | --- | --- |
| Central/tail grid and joins | 1153 | .982 / .0282 / .0289 / .0365 |
| Mixed ordinary, exponent and final-boundary sweep | 2601 | .9993 / approximately 1 / approximately 1 / approximately 1 |
| Original-input phase only | 569 | Max absolute sine/cosine error divided by eps: .783 / .755 / .672 / .787 |

All sweeps passed accuracy and classification checks. The near-one ratios in
the boundary sweep are dominated by the half-subnormal rounding allowance,
not large relative error in ordinary values. A naive native phase failed on
540/556/558/561 of the 569 phase samples. These are observed maxima, not a
uniform proven error bound.

The shared precision registry adds `AiryIdentities` and `AiryAccuracy` for
configured native types and MPFR128/256. Tests include Ai(0), the differential
equation, negative zeros, joins, huge negative neighbors, positive underflow
neighbors and increasing-precision accuracy. MPFR-independent identities and
actual missing-provider executables also run without MPFR. The aggregate
MPFR-special-function probe remains MPFR-specific; native families have their
own named probes.

Targeted validation passed 77 tests with MPFR enabled and 12 native-only tests,
plus the missing-provider executables and explanatory example. Replacing the
phase reducer by a naive native phase makes all four native accuracy probes
fail; removing the positive-tail guard makes the float32 probe fail. Both
mutations leave the MPFR probes passing. The 569-sample phase sweep also passed
AddressSanitizer and UndefinedBehaviorSanitizer at each native precision.
A combined Clang19 preflight of the seven new special-function families passed
116 core/numerical tests, including Airy at native32/64/80 and MPFR128/256;
native128 was disabled in that configuration. It compiled the current headers
against compatible previously built core libraries, so it remains targeted
compiler evidence. Full unfiltered
GCC/Clang, examples, CLI and Python configurations remain the publication gate;
these targeted checks do not establish that integration result.

Representative GCC13 -O2 costs, in microseconds per call on the validation
host, were:

| Region | float32 | float64 | float80 | float128 |
| --- | --- | --- | --- | --- |
| Central x=1 | 26 | 38 | 286 | 2015 |
| Negative asymptotic x=-40 | .81 | 1.24 | 1.71 | 10.7 |
| Positive final-range guard | 62 | 118 | 782 | 6123 |
| Negative argument near maximum exponent | 3.7 | 68 | 11568 | 11931 |

The four-word central method and extreme phase reduction trade speed for
precision preservation. Measurements are indicative, not universal performance
guarantees; no native Ai provider was being replaced. A faster central
approximation can be investigated separately against the same acceptance table.
