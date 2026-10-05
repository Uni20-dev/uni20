# Native log-Gamma and integer-order Bessel functions

`<uni20/core/math.hpp>` exposes `math::lgamma`, `math::lgamma_sign`,
`math::bessel_j`, and `math::bessel_y` for native real types when a suitable
platform routine is available. These adapters add no library dependency and
do not use Boost or MPFR to evaluate native arguments.

```cpp
#include <uni20/core/math.hpp>

// On a platform with the double-precision providers:
auto gamma = uni20::math::lgamma_sign(-0.5);
// gamma.value = log(abs(Gamma(-0.5))), gamma.sign = -1
auto logarithm = uni20::math::lgamma(-0.5); // Same value, without the sign.
auto j = uni20::math::bessel_j(3, 1.0);     // J_3(1)
auto y = uni20::math::bessel_y(-3, 1.0);    // Y_-3(1) = -Y_3(1)
```

## Availability and providers

CMake compiles and links probes independently for `float`, `double`,
`long double`, and configured binary128. The native result retains the input's
type. Unsupported signatures are constrained out; a missing extended-precision
routine never falls back to double. Generic callers can use `std::invocable`
on the corresponding `uni20::math` callable. These APIs accept native real
arguments, not conversion-only classes or an added `Precision` argument.

| Scalar | Log-Gamma | Bessel J/Y |
| --- | --- | --- |
| `float` | `lgammaf_r` | `jnf`, `ynf` |
| `double` | `lgamma_r` | `jn`, `yn` |
| `long double` (including native float80) | `lgammal_r` | `jnl`, `ynl` |
| Configured float128 | `lgammaf128_r` | `jnf128`, `ynf128` |

The last row uses the platform's binary128 C math routines where available.
It does not require those symbols on every platform with binary128 storage.
The generated capability macros are `UNI20_HAS_LGAMMA_R_{FLOAT,DOUBLE,LONG_DOUBLE,FLOAT128}`
and `UNI20_HAS_BESSEL_{FLOAT,DOUBLE,LONG_DOUBLE,FLOAT128}`. Each Bessel capability
requires both J and Y. Capability detection also applies to macOS; this is not
a claim that each macOS SDK supplies every routine in the table.

Log-Gamma uses only reentrant routines returning the sign through an output
parameter. Neither interface reads or modifies the global `signgam` variable.
The ordinary non-reentrant `lgamma` routine is not a fallback.

## Numerical contracts

`lgamma(x)` means the natural logarithm of the absolute Gamma value.
`lgamma_sign(x)` returns `lgamma_result<R>{value, sign}`. The sign is -1 or +1
where defined, zero for NaN, negative infinity and negative integer poles.
At positive/negative zero the sign is +1/-1, respectively. The value follows
the native provider, including infinities at poles and NaN propagation.

Bessel order comes first and must be an integer source other than `bool`.
The common signed-64-bit argument check is followed by provider lowering:
the absolute order must fit `int`, otherwise `std::out_of_range` is thrown
before calling the provider. Negative orders use
`J_-n(x) = (-1)^n J_n(x)` and `Y_-n(x) = (-1)^n Y_n(x)`.
Native Bessel Y has the provider's real domain: negative real arguments give
NaN; the zero endpoint is a pole. The provider determines floating-point
exceptions, range behavior and signed zeros. The MPFR overloads retain their
existing signed-64-bit order domain and exact-value contracts.

Native results do not promise MPFR's correct rounding. Validation includes
Gamma recurrence and half-integer identities, Bessel parity and recurrence,
special values, checked order conversion, and preservation of global sign
state. The shared precision matrix measures log-Gamma and Bessel accuracy at
float32/64/80/128 and MPFR 128/256 bits where configured, against independent
identities/series and wider MPFR references. Unavailable provider combinations
are recorded in the coverage report rather than counted as passing probes.

The remaining native functions are tracked in
[issue #67](https://github.com/Uni20-dev/uni20/issues/67) and the
[scalar math coverage plan](scalar_math_coverage_plan.md).
