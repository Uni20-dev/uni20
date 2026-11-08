# Barnes-G magnitudes and real-shift ratios

`<uni20/core/log_barnes_g.hpp>` provides an immutable
`log_barnes_g_context<R>` for native real types and configured `mpreal`:

```cpp
uni20::log_barnes_g_context<double> barnes;
auto a = barnes.log_abs_barnes_g_right({0.5, 2.0});
auto r = barnes.log_abs_barnes_g_ratio_right({0.5, 2.0}, 0.5);
// a.value = log|G(0.5 + 2i)|
// r.value = log|G(1 + 2i)| - log|G(0.5 + 2i)|
```

The ratio forms the difference directly. At large imaginary arguments the
individual logarithms grow as `y*y*log(y)` whereas their real-shift difference
is much smaller. Separate evaluation followed by subtraction can lose that
difference, or overflow even when the paired result is representable. The
paired operation is useful for form factors; model-specific transition rates,
normalizations and integration remain in the consuming application.

## Domain and precision

Both operations require finite `Re(z)>0`; the ratio additionally requires a
finite real `h` and a positive, representable shifted real part. Negative
shifts reverse the paired difference. Zero shifts and the normalization
values `log|G(1)| = log|G(2)| = 0` return exact zeros at the requested precision.
These are real logarithms of magnitudes. No branch of complex `log G`,
reflection formula, left-half-plane continuation or general complex shift is
exposed.

Construct an MPFR context with, for example,
`log_barnes_g_context<mpreal>(Precision::bits(256))`. Inputs must be exact or
have the requested precision. Conversion from another finite precision is
explicit at the call site. Barnes arithmetic uses 64 internal guard bits;
its Gamma context adds its own 64 guard bits. All result scalars are rounded
back to the requested precision. Ambient precision and exponent range are
unchanged. Native contexts keep native arithmetic and work without MPFR.

Contexts own coefficients and a normalization constant, with no mutable
per-call scratch, and can be shared between Async tasks. Gamma-only callers
do not construct any Barnes state.

## Evaluation

The implementation uses the recurrence `G(z+1) = Gamma(z) G(z)`,
`G(1)=1`, and the asymptotic expansion in
[DLMF 5.17](https://dlmf.nist.gov/5.17). With `v=z-1`, the part independent
of the additive normalization is

$$
F(v)=\left(\frac{v^2}{2}-\frac1{12}\right)\log v
-\frac{3v^2}{4}+\frac v2\log(2\pi)
+\sum_{k=1}^{N-1}\frac{B_{2k+2}}{4k(k+1)v^{2k}}.
$$

Arguments are shifted until `v=z+s-1` lies within `|arg(v)|<=3*pi/4` and
the omitted-term bound meets a working-precision target. This allows a
slightly negative real part of `v` at large imaginary argument, avoiding
an unnecessary Gamma recurrence. The public domain remains `Re(z)>0`.
For `w=z+s`, the Gamma recurrence gives

$$
\sum_{j=0}^{s-1}\log\Gamma(z+j)
=s\log\Gamma(w)-\sum_{j=0}^{s-1}(j+1)\log(z+j).
$$

This needs one Gamma evaluation after shifting, rather than one per shift.
The constant is prepared once by evaluating the same expression at `z=1`;
no finite-digit approximation to the Glaisher constant limits MPFR accuracy.
The paired operation cancels this constant algebraically.

For small relative shifts, write `q=h/v`. The leading difference is

$$
F_0(v+h)-F_0(v)
=hv(\log v-1)+\frac{h^2}{2}\log v+h^2T(q),
\qquad F_0(v)=\frac{v^2}{2}\log v-\frac{3v^2}{4},
$$

$$
T(q)=\sum_{k\ge1}\frac{(-1)^{k+1}q^k}{k(k+1)(k+2)}.
$$

This retains the shift even when `z+h` rounds back to `z`, and avoids
forming `v*v` on the small-relative-shift path. The `log(1+q)` term is also
evaluated by a convergent series for `|q|<=1/8`, with geometric tail bounds.
Both series factor out `q` before testing convergence: the next power in
`log(1+q)/q` and `T(q)/q` is compared with working epsilon divided by 64.
Consequently a tiny `q` does not trigger an early stop just because the whole
answer is absolutely small. Multiplying the sums and tail bounds by `q`
restores their original scale. This controls the series relative to that
factor; it does not promise relative accuracy near a zero of the complete
paired result. The other asymptotic bounds and returned diagnostics remain
absolute quantities.
Larger shifts use a factored polynomial difference and a scaled logarithm
ratio. Reciprocal-power differences use a product recurrence rather than
subtracting two powers. Real sums use Neumaier compensation.

## Diagnostics and work limits

| Field | Meaning |
| --- | --- |
| `value` | Real log-magnitude or paired log-ratio |
| `truncation_bound` | Analytic omitted-series bound evaluated in ordinary arithmetic |
| `roundoff_indicator` | Heuristic based on working epsilon, intermediate magnitudes and final rounding |
| `terms` | Included inverse-power Barnes coefficients |
| `shifts` | Barnes recurrence steps for this evaluation |

The counts exclude context construction, Gamma's internal work and the
small-logarithm series. Diagnostics include their numerical contributions.
For the standalone operation, the stored normalization's diagnostics are
included as well.

The remainder combines the Gamma remainder from
[DLMF 5.11(ii)](https://dlmf.nist.gov/5.11.ii) with the remainder of the
Gamma-containing Barnes expansion in
[Nemes (2014), Eq. (1.10)](https://arxiv.org/html/1406.2535v2).
If `N` is the first omitted inverse-power index, put
`c = B_(2N+2)/((2N+2)(2N+1))` and `s = sec(arg(v)/2)`. A bound on the
combined omitted series is

$$
\frac{|c|}{|v|^{2N}}
\left[s^{2N+2}+\frac{s^{2N+1}}{2N}\right].
$$

It is evaluated in logarithms with scaled components. The paired result
includes both remainders, the paired logarithm tails and any Gamma recurrence
error. Neither this ordinary-arithmetic bound nor the roundoff heuristic is
an outward-rounded interval certificate. In particular, a small truncation
bound cannot certify a small relative error near a zero. Truncation values
can underflow to zero. Arithmetic or diagnostics outside the scalar range
throw rather than reporting convergence; the implementation does not promise
every mathematically representable answer when intermediate products overflow.

`log_barnes_g_options` defaults to 4,096 recurrence steps and 512 term slots.
Native coefficients reuse the 16-slot Stirling table. MPFR coefficients are
generated from exact Bernoulli rationals, with a precision-dependent count;
the internal Gamma context has its own count at its guarded precision. The
last slot bounds the omitted series. The term budget also limits the
small-logarithm series. Insufficient coefficient budgets are rejected with
`std::invalid_argument`; exhausted recurrence/series budgets throw
`std::runtime_error`. Normalization is prepared during construction, so an
insufficient recurrence budget can fail at construction. Invalid domains or
precision mismatches throw `std::invalid_argument`; unrepresentable
arithmetic or diagnostics throw `std::overflow_error`.

Successful construction establishes the coefficient and normalization budgets;
it does not guarantee that every argument's series fits the per-call budget.
For example, at 2,000-bit output the default 512 slots prepare the context but
the real-shift call `z=2`, `h=30` exhausts its small-shift series. Constructing
the context with `{.max_terms = 1024}` permits that evaluation at the same
requested precision. The failure is explicit and returns no partial value.
Applications can retry with a larger work budget separately from increasing
precision. This distinction also applies to the internal Gamma context.

## Validation

The shared numerical registry covers all configured complex scalar families,
with recurrence, conjugation, negative shifts, output precision, large-argument
paired differences and independent reference checks. Fixtures are produced
by `tests/core/generate_log_barnes_g_references.py` using mpmath at 600 decimal
digits and retain 400 digits; mpmath is not a runtime or test dependency.
Dedicated MPFR tests cover 128, 256, 333, 512 and 1,024 bits. Tiny-shift
fixtures at `z=2+3i`, `h=+/-2^-200` detect premature absolute stopping even
when the shift itself was retained. Additional fixtures approach the domain
boundary from positive real parts and by negative shifts. Native-only tests
also check cases where separate evaluation loses the answer or overflows.
An Async test shares immutable contexts at two precisions and verifies
deterministic values without changing MPFR precision or exponent settings.
A 2,000-bit failure-and-retry test checks the work-budget contract against the
independent finite product `G(32)/G(2) = product(k!, k=1,...,30)`.
