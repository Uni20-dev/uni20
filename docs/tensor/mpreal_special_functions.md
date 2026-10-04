# MPFR special functions and elementary extensions

Include `<uni20/core/math.hpp>` and enable `UNI20_ENABLE_MPFR`. The real
functions below accept `mpreal` through both `uni20::name(...)` and the generic
`uni20::math::name(...)` interface. They require MPFR 4.2 or newer, the same
minimum as the numerical utilities. This surface does not add MPC overloads.

Native float32/64/80/128 use their native implementations for `tgamma`, `erf`
and `erfc`. The other functions on this page currently have no native adapter;
their native implementations are deferred. Enabling MPFR never supplies an
MPFR-backed implementation for a native scalar. Generic code can check
availability with `std::invocable` on the `uni20::math` callable.

## Precision and arithmetic state

Without an override, a unary approximate result retains its input precision.
Binary approximate operands must have matching precisions; an exact operand
uses the other operand's precision. All-exact inputs either produce a supported
exact rational result or throw `std::logic_error` when approximation is needed.
Unset operands throw; no default or thread-local precision is changed.

Except for `compound`, a trailing finite `Precision p` converts the numerical
inputs to `p` first, then evaluates at `p`, matching the existing elementary
contract. This can change an exact rational input before the function is
evaluated. Results always remain approximate in this mode, even at identities.
`Precision::exact()` is rejected as an override. Use the overload without an
override to retain exact results.

`compound(x,n,p)` follows the arithmetic-utility contract: it rounds the result
from the original input, without first rounding `x` or `1+x` to `p`. Its exact
rational path reuses the integer-power machinery. As with `pown`, an exact base
outside MPFR's configured exponent range can throw `std::overflow_error`
instead of changing that range.

```cpp
auto p = uni20::Precision::bits(256);
uni20::mpreal x("1/3", uni20::Precision::exact());
auto a = uni20::math::digamma(x, p); // Convert x to 256 bits, then evaluate.
auto b = uni20::math::compound(x, 3, p); // Round the rational result 64/27.
```

## Special functions

| Function | Meaning |
| --- | --- |
| `tgamma(x)` | Real Gamma |
| `lgamma(x)` | Natural logarithm of the absolute Gamma value |
| `lgamma_sign(x)` | Owning `lgamma_result<mpreal>{value, sign}` |
| `digamma(x)` | Logarithmic derivative of Gamma |
| `beta(a,b)` | Real Beta |
| `upper_gamma(a,x)` | Upper incomplete Gamma; `a` is the shape and `x` the lower integration endpoint |
| `erf(x)`, `erfc(x)` | Error function and its complement, each evaluated directly |
| `zeta(x)` | Riemann zeta |
| `expint(x)` | Real exponential integral Ei, with principal-value meaning |
| `dilog_real(x)` | Real part of Li2, including the continuation for real `x>1` |
| `bessel_j(n,x)`, `bessel_y(n,x)` | Integer-order Bessel functions of the first and second kinds; order comes first |
| `airy_ai(x)` | Real Airy Ai |
| `agm(a,b)` | Arithmetic-geometric mean |
| `factorial(n,p)` | Nonnegative integer factorial; explicit arithmetic state required |

Bessel orders and factorial arguments accept checked signed-64-bit integer
sources, excluding `bool`. Negative factorial arguments throw
`std::domain_error`; integer sources outside that range throw
`std::out_of_range`. Factorial is a factory: unlike the numerical overrides,
its required `p` may be `Precision::exact()`.

`lgamma_sign` returns `sign` equal to -1 or +1 where the Gamma sign is defined.
It reports zero for NaN, negative infinity and negative integer poles. At
approximate signed zero its sign follows the zero. `lgamma` always means
log-absolute-Gamma, so negative Gamma values do not by themselves produce NaN.

The wrappers retain MPFR's domains and exceptional-value behavior. In
particular, Gamma and digamma have NaNs at negative integer poles; Bessel Y
returns NaN for negative real arguments. The dilogarithm returns a real part,
not a complex branch value. Beta and upper incomplete Gamma retain provider
limitations involving internal range and work; Airy Ai is intended for
moderate arguments, typically magnitude below 500. See the
[MPFR transcendental reference](https://www.mpfr.org/mpfr-4.2.1/mpfr.html#Transcendental-Functions)
for the underlying routines.

## Elementary extensions

| Functions | Meaning |
| --- | --- |
| `exp10(x)` | Ten to the power x |
| `exp2m1(x)`, `exp10m1(x)` | Base-two/base-ten exponential minus one |
| `log2p1(x)`, `log10p1(x)` | Base-two/base-ten logarithm of one plus x |
| `sinpi(x)`, `cospi(x)`, `tanpi(x)` | Trigonometric functions of pi times x |
| `asinpi(x)`, `acospi(x)`, `atanpi(x)`, `atan2pi(y,x)` | Principal inverse trigonometric functions divided by pi |
| `sec(x)`, `csc(x)`, `cot(x)` | Reciprocal cosine, sine and tangent, with x in radians |
| `sech(x)`, `csch(x)`, `coth(x)` | Reciprocal hyperbolic cosine, sine and tangent |
| `compound(x,n)` | `(1+x)^n`, for `x>=-1`, with a checked signed-64-bit integer exponent |

All approximate paths call the dedicated provider routines, including the
reciprocal functions. The small-argument routines do not form a rounded
exponential-minus-one or logarithm-of-sum. Pi-scaled routines do not multiply
by a rounded pi before argument reduction.

`compound` follows its provider domain even for integer exponents: below -1,
approximate evaluation returns NaN and exact evaluation throws
`std::domain_error`. At -1, negative exponents produce approximate positive
infinity, but throw in exact arithmetic. Exponent zero gives one except below
the domain boundary. Unset operands remain errors even for exponent zero.

Approximate pi-scaled functions preserve provider signed-zero conventions.
For example, `sinpi` and `tanpi` preserve the sign of zero, while `cospi` at a
half-integer returns positive zero. Inverse functions have their usual real
domains. `atan2pi` follows quadrant and signed-zero rules; two exact zeros
have no direction and throw `std::domain_error`.

## Supported exact results

| Operation | Exact cases |
| --- | --- |
| Gamma/factorial | Positive integer Gamma, nonnegative integer factorial |
| Log-Gamma | Zero at inputs 1 and 2; sign +1 |
| Beta | Positive rational arguments with at least one positive integer |
| Upper Gamma | Zero endpoint when complete Gamma has an exact result |
| Zeta | All nonpositive integers, using exact Bernoulli numbers for negative odd arguments |
| Error/dilogarithm | `erf(0)=Li2(0)=0`, `erfc(0)=1` |
| Bessel J | `J_0(0)=1`, all other integer orders at zero give zero |
| AGM | Equal nonnegative arguments, or a zero and a nonnegative argument |
| `exp10`, `exp2m1`, `exp10m1` | Integer exponents |
| `log2p1`, `log10p1` | When `1+x` is a rational integer power of the corresponding base |
| `sinpi`, `cospi` | Rational results 0, +/-1/2, +/-1, using exact periodic reduction |
| `tanpi` | Quarter-integer arguments with finite rational result: 0 or +/-1; half-integer poles throw |
| `asinpi`, `acospi` | Inputs 0, +/-1/2, +/-1 |
| `atanpi` | Inputs 0, +/-1 |
| `atan2pi` | Nonzero axes and diagonals |
| `sec`, `sech` | One at zero |
| `compound` | Rational inputs in the domain and integer exponents, except division by zero |

These rules are deliberately explicit: the type does not perform general
symbolic simplification or prove arbitrary special-function identities.
Exact factorials and powers can require large storage; the exact negative-odd
zeta recurrence takes quadratic work in the order and grows rational operands.
There is no small artificial order cutoff. Request finite precision when a
large exact integer/rational result is not needed.

## Constants and examples

`pi<mpreal>`, `log_two<mpreal>`, `euler_gamma<mpreal>` and `catalan<mpreal>` are
symbolic descriptors. `.at(p)` evaluates them at finite precision. Arithmetic
with an approximate `mpreal` infers that operand's precision; arithmetic with an
exact operand cannot infer a working precision and throws. They are not stored
approximations and remain separate from exact rational constants.

Run `mpreal_special_example` for annotated examples of exact results,
log-Gamma signs, stable small arguments, pi-scaled argument reduction, result
rounding and constants. The shared numerical matrix tests native Gamma/error
functions where available and MPFR-specific operations at 128 and 256 bits.
Missing native operations are reported as unsupported, not as successful or
skipped MPFR-backed tests. See [numerical testing](../development/numerical_testing.md).
