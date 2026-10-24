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
Future binary128 implementations may also serve the other native types after
accuracy and performance validation; see the
[shared native implementation plan](scalar_math_coverage_plan.md#shared-native-implementations).
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

## Pi-scaled trigonometric functions

`math::sinpi`, `cospi`, `tanpi`, `asinpi`, `acospi`, `atanpi` and
`atan2pi` support native float32/64/80 and configured binary128, as well as
the existing MPFR scalar path. The forward functions evaluate at pi times the
argument; the inverse functions return angles divided by pi. Native unary
results retain the argument type. `atan2pi(y,x)` uses the common native floating
type, subject to the usual facade check that it preserves both operands'
precision and exponent range. Integer and conversion-only inputs are not
accepted. As with the existing elementary math interface, a trailing
`Precision` is an MPFR operation, not a native overload.

```cpp
auto phase = uni20::math::sinpi(0.5L); // exactly 1, without forming pi/2
auto turns = uni20::math::atan2pi(1.0, -1.0); // exactly 0.75
```

CMake checks the linked C23 forward family (`sinpi`, `cospi`, `tanpi`) and
inverse family (`asinpi`, `acospi`, `atanpi`, `atan2pi`) separately for each
native type. It requires exact signatures, including `_Float128` signatures
for binary128. The capability macros are `UNI20_HAS_PI_TRIG_{FLOAT,DOUBLE,LONG_DOUBLE,FLOAT128}`
and `UNI20_HAS_PI_INVERSE_{FLOAT,DOUBLE,LONG_DOUBLE,FLOAT128}`. A complete detected
family uses its libc provider. A missing or incomplete family uses a portable
implementation at the original precision. No newer libc is required, and
MPFR is never used to evaluate native arguments.

The portable forward implementation uses `remquo(x, 0.5)` to recover the
quadrant and an exact binary remainder in [-1/4,1/4]. It multiplies only that
remainder by the typed pi constant before calling native `sin`, `cos` or
`tan`. This preserves phase for large arguments, avoids overflow in `2*x` or
`pi*x`, and retains relative accuracy near integer zeros and half-integer
poles. The inverse implementation scales the native inverse function, with
explicit handling of exact endpoints, axes and diagonals. `atan2pi` uses
`atan2`, avoiding an overflowing or underflowing intermediate `y/x`.

The native value conventions follow IEEE pi-scaled arithmetic:

- At every integer, `sinpi` and `tanpi` return zero with the input's sign.
- `cospi` returns positive zero at every half-integer.
- `tanpi(n+1/2)` returns positive infinity for even n and negative infinity
  for odd n; at quarter-integers its finite values are exactly +1 or -1.
- Forward functions return NaN for infinite or NaN inputs. `asinpi` and
  `acospi` have domain [-1,1], with NaN outside it.
- `atanpi` maps infinities to +/-1/2. `atan2pi` preserves signed-zero
  quadrants, including both zero arguments, and handles infinite axes and
  diagonals. NaN in either argument yields NaN.

The MPFR path retains its provider conventions. In particular, MPFR 4.2.1
uses an alternating zero sign for `tanpi` at odd integers; native results follow
the input sign, as described in the current MPFR reference.

See the [MPFR function reference](https://www.mpfr.org/mpfr-current/mpfr.html#Transcendental-Functions)
and the GNU libc [forward](https://sourceware.org/glibc/manual/latest/html_node/Trig-Functions.html)
and [inverse](https://sourceware.org/glibc/manual/latest/html_node/Inverse-Trig-Functions.html)
interfaces. Native implementations do not promise correct rounding or a
portable `errno`/floating-point-exception contract. Their accuracy depends on
the native elementary provider and rounding of the typed pi constant. Tests
use round-to-nearest with gradual underflow, require errors within four output
grid spacings on a deterministic adversarial sample, and separately check
improvement with precision using analytic identities and a convergent series.
This is numerical validation, not a proven global error bound.

The shared precision matrix tests both the selected public path and the
portable implementation explicitly. Thus a successful fallback does not count
as validation of an unavailable libc provider. Core boundary tests run without
MPFR; the wider 512-bit oracle checks are reported unavailable in that build.

### Numerical acceptance

The following acceptance criteria apply to configured native float32/64/80/128
under round-to-nearest with gradual underflow. The reference is the function at
the exact represented input, not at an intended real input before rounding.
`PiScaledAccuracy` and `PiScaledPortableAccuracy` cover the public and portable
paths separately in `tests/numerics/test_pi_math_precision.cpp`.

| Region | Measure and intended criterion | Guarantee or evidence |
| --- | --- | --- |
| Integer/half-integer grids; quarter-integer tangent; inverse endpoints, axes and equal-magnitude diagonals | Exact value and specified zero/infinity sign | Value contract; core tests plus precision-matrix identities |
| Ordinary non-dyadic sample (`i/37`, -128 <= i <= 128); both neighbors of quarter integers from -4 to 4 | Absolute error <= four output grid spacings against 512-bit MPFR | Empirical acceptance target, not a domain-wide bound |
| Large arguments with representable quarter offsets; native minimum normal/subnormal multiples and largest finite values | Same absolute grid-spacing target, with exact checks for zeros and nonfinite results | Tests phase retention and exponent range separately from significand precision |
| Selected nonzero analytic values at 1/8, 1/2 and 1/4 | Four-spacing target and error at least sixteen times smaller than the preceding precision's rounded reference | Precision-improvement evidence from radical/rational identities and an independent atan series |
| Out-of-domain and nonfinite arguments | NaN classification, or the specified finite/infinite value and sign | Value contract, independently checked in core tests |

For a finite nonzero result `r`, an output grid spacing is
`next_up(abs(r)) - abs(r)`, evaluated in the wider reference type. This also
gives an absolute measure in the subnormal region. No relative error is used
at zeros; exact-zero signs have separate assertions. Pole-neighbor tests assess
evaluation at the represented input, not the conditioning of an unrounded
mathematical argument.

The recorded GCC 13 and Clang 19 runs satisfy these thresholds for their
configured portable paths. The tests do not record a per-function observed
maximum, so no tighter measured bound is claimed. A provider unavailable on a
test host has no numerical evidence from that run, even if its wiring is
checked synthetically. Accuracy on a new libc provider must be measured on
that provider before calling it validated.

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
