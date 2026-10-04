# Shared scalar math dispatch

The host scalar-math interface lives in `<uni20/core/math.hpp>`, in namespace
`uni20::math`. It supplies one place for native overload selection and
class-specific argument-dependent lookup (ADL). Algorithms no longer need
Krylov-private math wrappers or their own copies of the lookup idiom.

```cpp
auto magnitude = uni20::math::abs(z);
auto root = uni20::math::sqrt(x);
auto tolerance = uni20::math::sqrt(uni20::numeric_limits<Real>::epsilon(x));

// Exact input, explicit approximation at finite working precision.
auto approximate_root = uni20::math::sqrt(uni20::mpreal{2}, p);
```

## Operations

Each operation below is a stateless callable object. Arguments are read through
const references. Availability depends on the selected scalar implementation;
providing a name does not synthesize a missing scalar operation.

| Operations | Contract |
| --- | --- |
| `abs`, `sqrt`, `cbrt`, `pow`, `hypot` | Magnitude, roots and powers with the scalar implementation's domain and precision rules |
| `exp`, `exp2`, `expm1`, `log`, `log2`, `log10`, `log1p` | Exponentials and logarithms; dedicated small-argument operations avoid cancellation |
| `sin`, `cos`, `tan`, `asin`, `acos`, `atan`, `atan2` | Trigonometric functions in radians |
| `sinh`, `cosh`, `tanh`, `asinh`, `acosh`, `atanh` | Hyperbolic functions |
| `pown`, `rootn` | Signed 64-bit integer powers and real roots, accepting checked integer source types |
| `floor`, `ceil`, `trunc`, `round`, `round_even` | Integer rounding; ties away for `round`, ties to even for `round_even` |
| `fma`, `fdim`, `fmin`, `fmax`, `copysign` | Fused product/sum, positive difference, extrema, sign transfer |
| `fmod`, `remainder`, `remquo` | Truncating/nearest-even quotient rules; named remainder and signed low three quotient bits |
| `frexp`, `modf`, `ilogb`, `ldexp`, `scalbn` | Named decomposition results, exponent extraction and binary scaling |
| `sincos`, `sinhcosh` | Named owning pairs at one input precision |
| `nextafter`, `next_up`, `next_down` | Adjacent values on the first operand's grid |
| `isfinite`, `isnan`, `isinf`, `signbit` | Real classification with boolean results |
| `real`, `imag` | Read components by value, never expose mutable component references; real/integral values retain their type, and imaginary zero retains runtime precision |
| `conj` | Conjugate complex values; preserve the type of real and integral values |
| `tgamma`, `erf`, `erfc` | Native or MPFR provider evaluation |
| `lgamma`, `lgamma_sign`, `digamma`, `beta`, `upper_gamma`, `zeta`, `expint`, `dilog_real`, `bessel_j`, `bessel_y`, `airy_ai`, `agm`, `factorial` | MPFR special functions; native implementations deferred |
| `exp10`, `exp2m1`, `exp10m1`, `log2p1`, `log10p1` | MPFR base-ten and cancellation-safe extensions |
| `sinpi`, `cospi`, `tanpi`, `asinpi`, `acospi`, `atanpi`, `atan2pi` | MPFR pi-scaled trigonometry and inverse functions |
| `sec`, `csc`, `cot`, `sech`, `csch`, `coth`, `compound` | MPFR reciprocal functions and compound integer powers |

`abs_squared(x)` is a composed function: compute `abs(x)` once and square it.
It adds neither overflow scaling nor extra precision. The root-level
`uni20::isfinite(x)` interface retains its existing complex rule; the new facade
classification entries cover real scalars.
Mutable component access through the existing root-level helpers is separate
from the value-reading `math::real` and `math::imag` interface.

See [MPFR special functions and extensions](mpreal_special_functions.md) for
argument order, named results, domains and exact/finite precision contracts.

The interface currently targets host execution. Scalar-math support does not
imply a BLAS/LAPACK provider, a projected Krylov solver, or CUDA support.

## Lookup boundary

For class scalars, the dispatcher calls an unqualified operation from an
isolated helper scope. It does not import standard floating-point overloads
there. Scalar overloads can be free functions in the associated namespace or
hidden friends, including declarations introduced after `math.hpp`.

A deleted variadic fallback matches arguments without conversion. It prevents
an unfamiliar class from acquiring support merely through `operator double()`,
even when a base class or template argument makes `std` an associated namespace.
More specific scalar overloads, including constrained templates, outrank the
fallback. Do not implement an extension solely as an equally generic variadic
catch-all: provide an overload specific to the scalar family.

Native arithmetic uses the corresponding standard overload set or a typed
adapter for Uni20 result shapes and nonstandard operations. For real
arguments, the result's significand and maximum exponent range must be at least
as wide as every real operand's; a missing extension overload cannot be filled
by narrowing. Integer promotions follow the selected standard operation.
Native float32/64/80/128 operations do not use MPFR adapters. A function without
a selected native implementation remains unavailable for those types even
when MPFR is enabled; adding its native implementation is separate work.
For example, `sqrt(4)` returns `double`, whereas `abs(-4)` returns `int`.
`real`, `imag` and `conj` follow their explicit type-preserving rules above.
`copysign` and `nextafter` deliberately retain the first operand's type; sign
and direction operands are not precision budgets. Classification returns `bool`,
exponent extraction returns `int64_t`, and named real result components preserve
the input type. These result categories have separate dispatch checks.

A compiler extension such as `float128` needs explicit native support:
a type alias does not give the underlying fundamental type an associated
namespace. The configured overload must preserve its precision. See the C++
[ADL rules](https://eel.is/c++draft/basic.lookup.argdep).

Unsupported calls are excluded by constraints, so `requires` and
`std::invocable` can check availability. ADL implementations control their own
result type; registering a scalar trait is not required just to supply an
operation. Overload availability does not prove the implementation is correct.

## Precision, exceptions and value semantics

Scalar overloads retain their [exactness and precision policy](scalar_policy.md).
An exact result stays exact when the operation supports it. An all-exact
operation requiring approximation needs explicit finite precision. Trailing
`Precision` arguments are forwarded to supported scalar overloads; the
dispatcher never invents precision, changes defaults or ignores an unsupported
precision argument.

For example, `math::sqrt(mpreal{4})` returns exact two;
`math::sqrt(mpreal{2})` throws; and `math::sqrt(mpreal{2}, p)` approximates at `p`.
`math::sqrt(2.0, p)` is not a supported call. MPFR rounding, fused arithmetic,
selection, remainders, decomposition, scaling and integer powers/roots with a
trailing precision round the result from the original operands. Elementary and
paired functions retain their input-conversion contract. Neighbor operations
first select a grid. See the [utility contracts](mpreal.md#arithmetic-and-numerical-utilities)
for these distinctions and exceptional values.

Native `pown` uses repeated squaring without converting the integer exponent to
a floating type. Its rounding error can accumulate with the exponent; it is not
a correctly rounded provider routine. Native `rootn` uses `sqrt`/`cbrt` for
orders of magnitude two/three, and `pow(abs(x), 1/n)` otherwise, with explicit
real-domain and odd-sign handling. The latter includes rounding of `1/n` in the
native type. Native paired functions call their components independently.
These algorithms retain float80/float128 precision but do not imply MPFR's
correct-rounding guarantees. They do not use an optional MPFR build to change
the behavior of native types.

Ordinary dispatch retains the implementation's applicable `constexpr` and
`noexcept` properties and propagates exceptions. Reading a component by value
can additionally copy it; its exception specification includes that copy.
The facade does not introduce expression templates or retain its arguments.

## Why a separate namespace

Existing `uni20::sqrt(mpreal)`, `uni20::sqrt(mpcomplex)` and exact-constant
operations remain the scalar implementations found through ADL. A generic
forwarding function also named `uni20::sqrt` can itself become an ADL candidate
for Uni20 types and produce recursive lookup or constraints. A callable object
cannot share its namespace and name with those free functions.

`uni20::math::sqrt` separates algorithm calls from scalar overloads without
requiring explicit per-type registration. Importing `std::sqrt` into `uni20`
would extend the known overload set, but qualified root-level calls would still
not discover external overloads through ADL. Explicit per-type registration is
a viable alternative, but would require adapters for external scalar families
that already supply suitable math functions.

## Validation and extension

The shared [numerical precision matrix](../development/numerical_testing.md)
covers real and complex scalar math at each configured precision, using values
that expose narrowing. Core tests cover late hidden friends, templated
overloads, unsupported and conversion-only classes, exception propagation,
component value semantics and exact/explicit-precision operations.

New operations should use the same lookup boundary, with their own result and
precision contract and an independent numerical check. Before using the facade
in device execution, add the applicable host/device annotations and an actual
CUDA compilation probe; annotation alone does not establish device-callability.
MPFR/MPC scalar implementations remain host-only.

The [scalar math coverage plan](scalar_math_coverage_plan.md) tracks remaining
arithmetic, decomposition, classification, constants and special functions.
The elementary expansion covers real scalars; existing native complex and MPC
operations remain available only where those implementations provide overloads.
