# Native special functions and constants

`<uni20/core/math.hpp>` exposes `math::beta`, `math::zeta`, `math::expint`,
`math::lgamma`, `math::lgamma_sign`, `math::bessel_j`, and `math::bessel_y`
for native real types when a suitable standard-library or platform routine is
available. These adapters add no library dependency and do not use Boost or
MPFR to evaluate native arguments.

```cpp
#include <uni20/core/math.hpp>

// On a platform with the double-precision providers:
auto gamma = uni20::math::lgamma_sign(-0.5);
// gamma.value = log(abs(Gamma(-0.5))), gamma.sign = -1
auto logarithm = uni20::math::lgamma(-0.5); // Same value, without the sign.
auto j = uni20::math::bessel_j(3, 1.0);     // J_3(1)
auto y = uni20::math::bessel_y(-3, 1.0);    // Y_-3(1) = -Y_3(1)
```

## Standard-library special functions

When `<cmath>` advertises `__cpp_lib_math_special_functions >= 201603L`,
Uni20 exposes these adapters for `float`, `double` and `long double`:

| Uni20 callable | Standard provider | Domain and interpretation |
| --- | --- | --- |
| `math::beta(x,y)` | `std::beta(x,y)` | Both arguments positive; same native real type |
| `math::zeta(x)` | `std::riemann_zeta(x)` | Real Riemann zeta; pole at one |
| `math::expint(x)` | `std::expint(x)` | Real principal-value Ei; pole at zero |

```cpp
auto b = uni20::math::beta(0.5L, 0.5L); // long double, approximately pi
auto z = uni20::math::zeta(2.0);        // double, approximately pi^2/6
auto e = uni20::math::expint(-1.0f);    // float, negative real Ei
```

The standard feature macro is sufficient to select these interfaces; there are
no redundant per-function CMake probes. The return type is the input type.
Integer arguments, mixed native types and conversion-only classes are not
accepted by these adapters. Convert explicitly when needed.

The standard library determines numerical accuracy, floating-point error
reporting and range behavior. In particular, native Beta does not promise
MPFR's extended real domain; the C++23 zeta contract leaves behavior below
-170 implementation-defined. These are provider wrappers, not a promise of
correct rounding or uniform behavior at every exceptional argument.
The existing MPFR overloads retain their own domains and precision rules.

If the macro is absent, these native overloads are unavailable. Configured
binary128 needs its own provider and is deliberately unsupported for these
three functions; a standard template declaration accepting an extension type
is not enough to establish that its implementation preserves precision.
The shared numerical tests cover native float32/64/80 and MPFR 128/256-bit
identities, and compare non-dyadic results with independent 512-bit references
to demonstrate increasing accuracy. Identity tests also run without MPFR.

### Global Gamma sign state

Application code using these adapters must not read, write, or rely on the
global `signgam` variable, including reading it after a non-reentrant platform
`lgamma` call. Use `math::lgamma_sign(x)` when the Gamma sign is needed; its
returned sign belongs to that invocation.

This restriction accommodates a known upstream limitation. The inspected
libstdc++ implementations of native Beta and zeta call ordinary `lgamma`, which
writes global `signgam` on glibc. Their numerical results do not consume that
sign output. Uni20 retains the standard-library adapters, but concurrent calls
on affected providers still have an internal write/write data race even when
application code never accesses `signgam`. The restriction does not make those
providers formally race-free, and race detectors can report the upstream
defect. Uni20 adds no serialization or replacement numerical algorithm here.
The same underlying issue is tracked for `std::poisson_distribution` in
[GCC bug 111726](https://gcc.gnu.org/bugzilla/show_bug.cgi?id=111726).

This limitation concerns the affected native Beta and zeta providers. Uni20's
own log-Gamma adapters use the reentrant routines described below.

## Typed constants

`<uni20/core/math_constants.hpp>` supplies `uni20::pi<Real>`,
`uni20::log_two<Real>` and `uni20::euler_gamma<Real>`. It is also included by
`math.hpp`. For native types these are `constexpr` values of type `Real`, using
`std::numbers::pi_v<Real>`, `ln2_v<Real>` and `egamma_v<Real>`, respectively.
This includes configured binary128 with the project's supported typed
standard constants. No value is constructed by widening a double constant.

```cpp
constexpr auto angle = uni20::pi<long double> / 3;
constexpr auto ln2 = uni20::log_two<double>;

// With MPFR enabled and mpreal.hpp included:
auto p = uni20::Precision::bits(256);
auto wide_pi = uni20::pi<uni20::mpreal>.at(p);
auto twice_pi = uni20::mpreal(2, p) * uni20::pi<uni20::mpreal>;
```

The MPFR specializations remain precision-aware descriptors, including
`catalan<mpreal>`. They require finite precision, explicitly or from an
approximate operand. Native Catalan, integral and complex constant types are
not provided. Constants are validated against wider MPFR values in the shared
precision matrix, including binary128 where configured.

## Log-Gamma and Bessel providers

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
