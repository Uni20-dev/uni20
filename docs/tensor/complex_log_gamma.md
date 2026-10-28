# Complex log-Gamma and real-shift ratios

`<uni20/core/complex_log_gamma.hpp>` provides an immutable
`log_gamma_context<R>` for native real types and configured `mpreal`. Its
concrete consumer is a scattering-phase kernel requiring the phase of a
Gamma ratio at large imaginary arguments. Evaluating the two large log-Gamma
phases separately can destroy their order-one difference.

```cpp
uni20::log_gamma_context<double> gamma;
auto a = gamma.log_gamma_right({0.5, 2.0});
auto b = gamma.log_gamma_ratio_right({0.5, 2.0}, 0.5);
// b.value = analytic log Gamma(1 + 2i) - analytic log Gamma(0.5 + 2i)
```

With MPFR enabled, construct `log_gamma_context<mpreal>(Precision::bits(256))`.
Each input must be exact or have that finite precision; conversion from another
finite precision is explicit at the call site. Native contexts require their
type's precision. The implementation uses 64 internal guard bits for MPFR and
rounds the value and both diagnostics back to the requested precision. It
changes neither ambient precision nor the MPFR exponent range. Native callers
retain native arithmetic and require no MPFR dependency.

## Domain and branch

Both operations require finite components and `Re(z)>0`. The ratio also
requires a finite real shift `h`, a representable shifted argument, and
`Re(z+h)>0`. The analytic log-Gamma is real on the positive real axis and
continuous throughout this half-plane. Its imaginary part is not reduced
modulo $2\pi$; it is **not** the principal logarithm of a computed Gamma value.
The ratio uses this same branch. A zero shift gives zero at the requested
precision. Reflection, poles, general complex shifts, Barnes G and physical
normalization factors are outside this interface.

## Evaluation and diagnostics

The implementation shifts the argument rightward with the Gamma recurrence,
then applies the Stirling series in
[DLMF 5.11.1](https://dlmf.nist.gov/5.11.E1). It uses the complex remainder
estimate in [DLMF 5.11(ii)](https://dlmf.nist.gov/5.11.ii): the first omitted
term multiplied by $\sec^{2n}(\arg(z)/2)$. That estimate is evaluated in logs
with scaled components so an overflowing magnitude or underflowing power
does not incorrectly decide when to stop shifting.

Leading terms use `z*(log(z)-1)`, with the real component evaluated by fused
multiply-add so cancellation occurs before an overflowing product rounds.
Near the positive real axis, `x*atan(y/x)` is evaluated as `y` when its cubic
correction is below working roundoff. This preserves the phase when `y/x`
underflows. Roundoff accumulation is scaled by epsilon before taking
magnitudes; this also permits finite complex components whose Euclidean
magnitude exceeds the scalar range.

For the ratio, both arguments use a common recurrence, starting from the
smaller real part; negative shifts reverse the resulting difference. Reciprocal-power
differences are formed directly. For a small relative shift $w=h/z$, the
leading difference uses

$$
h\log z+(h-1/2)\log(1+w)
+h\left(\frac{\log(1+w)}{w}-1\right).
$$

The final parenthesis is evaluated as its own convergent series. This avoids
forming `z*log(1+w)-h` and retains the leading difference even if `z+h` rounds
back to `z`. Large relative shifts cancel the common binary scale before
forming differences of argument logarithms, preserving accuracy when both
arguments are tiny. Near the positive real axis, the paired leading phase is
evaluated as

$$
y\left[\log(1+h/x)+\frac{h}{2x(x+h)}\right].
$$

This first-order expression is used only when `abs(y/x) <= sqrt(epsilon)/2`,
so its omitted relative correction is below working roundoff. The product
`y*h/x` keeps its binary exponent separate until the final scaling; neither
`y/x` nor `h/x` needs to be representable separately. The ratio's
truncation diagnostic includes both Stirling remainders and the small-logarithm
series tails.

Every result contains:

| Field | Meaning |
| --- | --- |
| `value` | Complex analytic logarithm or log-ratio |
| `truncation_bound` | Analytic bound on omitted series, evaluated in ordinary arithmetic |
| `roundoff_indicator` | Heuristic based on accumulated term magnitudes, working epsilon and final rounding |
| `terms` | Number of included Stirling coefficients |
| `shifts` | Number of common recurrence steps |

Neither diagnostic is an outward-rounded interval certificate. In particular,
small truncation does not prove small floating-point error near a zero of the
answer. A tiny truncation estimate can underflow to zero while the roundoff
indicator remains nonzero. Accuracy claims require independent references,
not just these diagnostics.

The context stores `B_(2k)/(2k*(2k-1))`. Native evaluation shares the existing
16-coefficient table with Uni20's real Stirling helpers. MPFR coefficients come
from exact rational arithmetic at the working precision; the shared Bernoulli
recurrence also supplies exact zeta values. Its internal conventional output
has `B_1=-1/2`.

`log_gamma_options` defaults to 4,096 recurrence steps and 512 term slots.
Native evaluation uses at most 16 coefficient slots; MPFR selects
`floor(working_bits/6)+8`, including the first omitted coefficient. An
insufficient MPFR coefficient budget is rejected before coefficient generation.
The term budget also limits each small-logarithm series. Exhausting recurrence
or series work throws `std::runtime_error`; invalid inputs/precision throw
`std::invalid_argument`, and unrepresentable arithmetic or diagnostics throw
`std::overflow_error`. The algorithm does not silently lower precision or
return a partial result as converged.

## Validation

The shared numerical precision registry covers recurrence, conjugation,
explicit precision and independent complex references for all configured
complex scalar families. Generic reference fixtures are generated with
mpmath at 800 decimal digits and retain 400 digits; the generator is
`tests/core/generate_complex_log_gamma_references.py`. It is not needed to
run tests. Dedicated MPFR checks cover 128, 256, 333 and 1,024 bits.

Range regressions include tiny arguments, tiny imaginary parts, shifts smaller
than an argument ULP, and a ratio phase at very large imaginary arguments
where separate log-Gamma subtraction demonstrably fails. Componentwise phase
checks span all configured precisions, including native ratios whose
intermediate angle underflows. Finite results whose unfused leading product
overflows are checked across all configured native types against an MPFR
oracle. Work-budget, precision-mismatch and branch tests are separate from
numerical accuracy checks. The reusable context stores no per-call mutable
scratch.
