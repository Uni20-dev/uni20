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
| `ceil`, `ldexp` | Native rounding and power-of-two scaling; MPFR overloads remain planned |
| `real`, `imag` | Read components by value, never expose mutable component references; real/integral values retain their type, and imaginary zero retains runtime precision |
| `conj` | Conjugate complex values; preserve the type of real and integral values |

`abs_squared(x)` is a composed function: compute `abs(x)` once and square it.
It adds neither overflow scaling nor extra precision. Classification continues
to use the existing `uni20::isfinite(x)` interface, including its complex rule.
Mutable component access through the existing root-level helpers is separate
from the value-reading `math::real` and `math::imag` interface.

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

Native arithmetic uses the corresponding standard overload set. For real
arguments, the result's significand and maximum exponent range must be at least
as wide as every real operand's; a missing extension overload cannot be filled
by narrowing. Integer promotions follow the selected standard operation.
For example, `sqrt(4)` returns `double`, whereas `abs(-4)` returns `int`.
`real`, `imag` and `conj` follow their explicit type-preserving rules above.

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
`math::sqrt(2.0, p)` is not a supported call. Uni20 supplies no MPFR `ceil` or
`ldexp` scalar overload in the current implementation, so these calls with
`mpreal` are also rejected rather than converted to a native float.

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
