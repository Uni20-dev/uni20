# Arbitrary-precision real scalars

`uni20::mpreal` is an experimental eager value type that holds either an exact
GMP rational or a finite-precision MPFR approximation. Optional MPC supplies
`uni20::complex<mpreal>` with two exact rational components or one MPC payload.
Symbolic constants such as `pi<mpreal>` remain separate descriptor types.

This is CPU arithmetic. Tensor construction defaults and the optional
[MPLAPACK product/solve backend](../linalg/mplapack_mpfr.md) are described below.
Table schemas and CLI selection do not yet support arbitrary precision.

## Configuration

Install development packages, then enable the optional scalar layer:

```sh
sudo apt install libgmp-dev libmpfr-dev
cmake -S . -B build_codex/mpfr -DUNI20_ENABLE_MPFR=ON
cmake --build build_codex/mpfr --target uni20_mpfr_tests mpreal_example
ctest --test-dir build_codex/mpfr --output-on-failure -R 'MpReal'
```

MPFR 4.2 or newer and a thread-safe MPFR build are required. There is no
source-download fallback. `CMAKE_PREFIX_PATH` supports a non-system prefix;
individual paths can be supplied through `UNI20_GMP_INCLUDE_DIR`,
`UNI20_GMP_LIBRARY`, `UNI20_MPFR_INCLUDE_DIR`, and `UNI20_MPFR_LIBRARY`.
CMake checks compilation and linking, and checks MPFR thread-local support
when target executables can run. Scalar construction also checks thread-local
support, including in cross builds. Consumers linking `uni20_core` inherit the
required includes and libraries. With the option off, those dependencies are
not searched for or linked.

MPC is not needed for real-only scalars. Install `libmpc-dev` and set
`UNI20_ENABLE_MPC=ON` together with `UNI20_ENABLE_MPFR=ON` for complex scalars.
Custom installations accept `UNI20_MPC_INCLUDE_DIR` and `UNI20_MPC_LIBRARY`.
The MPFR scalar option is independent of
`UNI20_ENABLE_MPLAPACK`, which currently selects binary128 dense kernels.

## Explicit working precision and value semantics

```cpp
#include <uni20/core/mpreal.hpp>
using namespace uni20;
using namespace uni20::literals;

auto p = Precision::bits(256);
// Or Precision::decimal_digits(80): at least 80 decimal significand digits.
mpreal zero(p);
mpreal x{"0.1", p};
mpreal y{"0.2", p};
auto z = x + y;                   // Owning mpreal, evaluated immediately.
```

Default construction and integer construction produce exact values: `mpreal{}`
is exact zero and `mpreal{1}` is exact one. `mpreal{0.1_mp}` retains the exact
rational `1/10`. `Precision::exact()` identifies this state; `is_exact()` queries
it. `bit_count()` throws for exact precision, and precision values have equality
but no numerical ordering. `common_precision(a, b)` treats exact state as neutral
and rejects differing finite precisions.

```cpp
mpreal third = mpreal{1} / mpreal{3}; // Exact 1/3, no working precision.
mpreal sum{};
sum += third;
sum += third;
sum += third;                       // Exactly 1.
sum += mpreal{"0.1", p};             // Now approximate at p.
```

Exact arithmetic covers `+`, `-`, `*`, `/`, unary signs, `abs`, `conj`,
classification and comparison. `sqrt` preserves rational perfect squares;
`hypot` does likewise when the sum of squares has a rational square root.
Real `pow` supports integer exponents (including negative exponents) and rational
exponents whose real root is rational: `pow(16, 3/4)` is exactly 8. For negative
bases, exact rational powers use the real root when the denominator is odd.
This differs from approximate MPFR `pow`, where a noninteger approximate exponent
on a negative base produces NaN. Zero to the zeroth power is one.
`cbrt` preserves rational perfect cubes, including negative inputs;
`exp2` preserves rational powers of two. `log2` and `log10` return exact integers
when the rational input is an integer power of their base, including reciprocal
powers: `log2(1/8)` is exactly -3. These checks use integer arithmetic, without
first approximating the input.

Exact elementary identities are:

| Result | Exact-input cases |
| --- | --- |
| Zero | `log(1)`, `log2(1)`, `log10(1)`, `log1p(0)`, `expm1(0)`, `sin(0)`, `tan(0)`, `asin(0)`, `acos(1)`, `atan(0)`, `atan2(0, positive)`, `sinh(0)`, `tanh(0)`, `asinh(0)`, `acosh(1)`, `atanh(0)` |
| One | `exp(0)`, `exp2(0)`, `cos(0)`, `cosh(0)` |

These rules never reclassify approximate operands as exact.

Exact division by zero and negative powers of zero throw `std::domain_error`.
Approximate division retains MPFR's infinity/NaN behavior. Exact arithmetic can
grow large numerators and denominators; it is not automatically rounded to limit
cost. A result outside the supported rational representation requires explicit
finite precision: `sqrt(mpreal{2})` throws; use `sqrt(mpreal{2}.at(p))`.
This is rational arithmetic, not a symbolic algebra system.

Algorithms seeking an approximate solution, such as Newton iteration, must
require finite working precision from their inputs or an explicit precision
argument before iterating. Rational arithmetic alone supplies no error budget:
even Newton iteration for `sqrt(2)` can keep producing larger fractions without
ever reaching the desired irrational value. Exact identities inside such an
algorithm do not establish that its overall approximation policy is valid.
This is the policy for future runtime-precision algorithm support, not a claim
that the current Krylov or other generic iterative APIs support `mpreal`.

`uni20::is_exact(x)` in `core/numeric_limits.hpp` is the generic value query:
it calls a runtime scalar's member `is_exact()` or uses the static
`numeric_limits<T>::is_exact` property for ordinary types. Integers report true;
floating types report false, including `double{1}`. Unset runtime values report
false. `core/math.hpp` also supports ordinary complex values. The type-level
`numeric_limits<T>::is_exact` remains a compile-time property, not a runtime
query; `mpreal` is not an always-exact type. Do not interpret a floating value's
binary representability as an exact-arithmetic guarantee.

`scalar_like(exemplar, value)` in `core/math.hpp` constructs the exemplar's scalar
type with its precision, or uses ordinary construction for fixed-precision types:
`sqrt(scalar_like(x, 2))` works for both an approximate `mpreal x` and a `double x`.
An exact exemplar remains exact and cannot supply a finite approximation budget.

Explicit `mpreal{uninitialized}` produces an unset placeholder. Copying, moving,
swapping and assignment are safe, but numerical use of an unset value throws
`std::logic_error`. `initialized()` distinguishes it from exact zero. Explicitly
uninitialized tensor storage uses this state; ordinary `std::vector<mpreal>(n)`
contains exact zeros. `Precision::decimal_digits` uses a conservative conversion
to binary bits and can allocate slightly more bits than the mathematical minimum.

Copy and move construction and assignment preserve the source value and
exact/approximate state and precision. Moving leaves the source unset; self-move
assignment preserves the value. No arithmetic expression retains references to
its operands.

Binary arithmetic between approximate `mpreal` values, including compound
assignment, requires equal finite precision. A mismatch throws
`std::invalid_argument` before modifying either operand. Explicit conversion uses `x.at(p)` or `mpreal{x, p}`; reducing precision
rounds, and increasing precision cannot restore digits already lost. With one
exact operand, basic `+`, `-`, `*` and `/` retain its rational value until the
result is rounded at the approximate operand's precision. For example, at three
bits `mpreal(1, p) - mpreal{9.0_mp / 8_mp}` gives `-1/8`, rather than zero.
For finite operands and a nonzero divisor, the result is correctly rounded;
complex results round each component once. With two exact operands, supported
rational arithmetic stays exact. Assigning an
exact value makes the destination exact, independently of its previous precision.
An approximate result is never reclassified as exact, even when it equals an
integer; `.at(Precision::exact())` rejects approximate inputs.
Compound arithmetic supports self-aliasing. Two approximate operands update the
destination in place; exact arithmetic and state transitions may allocate a new
payload.

Most mixed real operations use MPFR's rational-operand routines. Exact divided
by approximate, and mixed complex multiplication/division, use exact rational
temporaries for the finite stored binary values, then round the result. Those
temporaries can grow with exponent magnitude as well as rational size. This
first implementation prioritizes accurate cancellation; a later implementation
may replace them with certified adaptive-precision evaluation. Approximate-only
operations continue to use MPFR/MPC directly. Transcendental functions and
explicit-precision math overloads retain their documented input-conversion
semantics; the single-rounding rule here applies to basic arithmetic operators.

```cpp
auto q = Precision::bits(400);
auto promoted = x.at(q);
x = promoted;                    // x now also has 400 bits.
// x + y would now throw: y still has 256 bits.
auto sum = x + y.at(q);
```

Arithmetic and conversions round to nearest, ties to even. Native floating-point
inputs require a constructor with explicit precision, such as `mpreal{0.1, p}`;
this imports the already-rounded binary64 value. Use decimal text or `_mp`
literals to avoid that initial rounding. Plain integer operands are accepted.
No implicit conversion to a native floating type is provided; explicit `double`
and `long double` conversions, plus explicit `float` conversion, accept exact
and approximate values. Exact conversion rounds directly to the destination
format, including subnormals, without an intermediate double-rounding error.
Borrowed MPFR access still requires `.at(p)` for exact values. `copy_to(mpfr_ptr)` explicitly rounds either state into an initialized
MPFR destination at the destination's precision.

## Exact literal arithmetic

```cpp
constexpr auto decimal = 0.12345678901234567890123456789_mp;
auto exact = 0.1_mp + 0.2_mp;      // Owning exact_constant equal to 3/10.
auto third = 1_mp / 3_mp;          // Owning exact_constant equal to 1/3.
auto a = exact.at(p);             // Round once to p.
auto b = x + third;               // Add exact third, then round at x's precision.
```

The `_mp` literal records source characters without conversion through a machine
float. Its descriptor can be `constexpr`; operations on descriptors materialize
owning GMP rationals at runtime. `exact_constant` accepts decimal or integer-fraction text for runtime
input and formats as a reduced numerator/denominator, or an integer.

Default construction of `exact_constant` produces zero. Move construction transfers
the payload without allocating and leaves the source unset; `initialized()` and
`is_exact()` then return false. Copying, moving, swapping and assignment support
unset states, and self-move assignment preserves the value. Numerical use of an
unset constant, including conversion to `mpreal`, throws `std::logic_error` until
it is assigned a value again. Moving an exact `mpreal` likewise requires no GMP
allocation.

Precisionless operations are unary signs, addition, subtraction, multiplication,
division, and exact comparisons. Division by zero throws `std::domain_error`.
There is no symbolic expression tree. Decimal point and decimal exponent notation
are supported; hexadecimal and binary spellings are rejected. Digit separators
between digits are accepted. All integer spellings are interpreted in base ten,
including those with leading zeros. Exact zero has no sign.

Decimal parsing limits the power-of-ten expansion to 1,000,000 places, after
combining the exponent with the number of fractional digits. Larger scales throw
`std::out_of_range` before expansion, including for zero inputs. This prevents
accidentally enormous allocations from short decimal spellings; it does not cap
`mpreal` working precision or the size of results from exact rational arithmetic.

Mixing an exact constant with `mpreal` in basic arithmetic rounds the result at
the real operand's precision. Each approximate operation is a rounding boundary:

```cpp
auto low = Precision::bits(3);
mpreal one(1, low);
auto together = one + (0.125_mp + 0.125_mp); // 1.25
auto separate = (one + 0.125_mp) + 0.125_mp; // 1: each tie rounds to even.
```

Comparisons do not round operands. Two `mpreal` values may be compared even if
precisions differ, and comparisons with an exact rational compare their actual
values. Thus `mpreal{"0.1", p} == 0.1_mp` is false: a finite binary value cannot
represent exactly 1/10.
Native integer comparisons also remain exact, even when the integer needs more
bits than the real's working precision. Signed and unsigned integers fitting
`long` and `unsigned long`, respectively, use MPFR's integer comparisons without
allocating rational temporaries; wider types retain an exact rational fallback.
Equality with integer zero uses a direct zero check. Both operand orders are
supported, signed zeros compare equal, and NaNs are unordered.

## Constants, elementary functions, and exceptional values

```cpp
auto pi_value = pi<mpreal>.at(p);
auto twice_pi = mpreal(2, p) * pi<mpreal>; // Infer p, evaluate immediately.
auto root = sqrt(mpreal{2}, p);            // Select an approximation for an exact input.
auto spacing = epsilon(p);                // 2^(1-p.bit_count()).
```

The pi descriptor computes pi at the requested precision, unlike promoting a
previously rounded approximation. It supports arithmetic with an existing
`mpreal`; precisionless combinations involving pi are unsupported. This slice
supplies `pi<mpreal>`, not a generic replacement for `std::numbers`.

The following real scalar functions are also available through
[`uni20::math`](scalar_math_design.md) for generic algorithms:

| Family | Functions |
| --- | --- |
| Magnitude, powers and roots | `abs`, `sqrt`, `cbrt`, `pow`, `hypot` |
| Exponentials and logarithms | `exp`, `exp2`, `expm1`, `log`, `log2`, `log10`, `log1p` |
| Trigonometry (radians) | `sin`, `cos`, `tan`, `asin`, `acos`, `atan`, `atan2` |
| Hyperbolic functions | `sinh`, `cosh`, `tanh`, `asinh`, `acosh`, `atanh` |

Each accepts an optional trailing `Precision`. Without it, the exact-result
rules above apply; approximate operands supply working precision, and differing
finite precisions are rejected.

An explicit finite `p` requests **input conversion followed by evaluation**:
`f(x, p)` means `f(x.at(p))`, and `f(x, y, p)` means
`f(x.at(p), y.at(p))`. Inputs remain unchanged. The result is always approximate
at `p`, even for `sqrt(mpreal{4}, p)` or `exp(mpreal{}, p)`. Binary overloads
therefore accept differing operand precisions. `Precision::exact()` and unset
operands throw `std::logic_error`; omit the precision argument to retain supported
exact results when the inputs are exact. No ambient precision is changed, including across async tasks.

```cpp
auto exact_root = sqrt(mpreal{4});       // Exact 2.
auto approximate = sqrt(mpreal{4}, p);   // Approximate 2 at p.
auto irrational = sqrt(mpreal{2}, p);    // Approximate sqrt(2) at p.
auto power = pow(x, y, p);              // Convert both inputs, then evaluate.
```

This does not promise one correctly rounded evaluation of the original rational
expression. For example, at two bits `sqrt(mpreal{6.4_mp}, p)` first rounds 6.4
to 6, then rounds its square root to 2. Evaluating the original square root and
rounding only the final result would produce 3. Approximate paths retain MPFR's
exceptional-value rules: `sqrt(mpreal{-1}, p)` yields NaN.

`expm1` and `log1p` call the dedicated MPFR routines. Use them for small
arguments: `exp(x)-1` and `log(1+x)` can lose the entire result even at the
chosen precision. Hyperbolic functions likewise call their MPFR routines.
For approximate real inputs, `asin`/`acos` outside [-1, 1], `acosh` below 1,
`atanh` outside [-1, 1], and `log1p` below -1 return NaN. `atanh(±1)` gives
signed infinity, and `log1p(-1)` gives negative infinity. `cbrt` has a real
negative branch. `expm1`, `log1p`, `cbrt`, `asin`, `sinh`, `tanh`, `asinh` and
`atanh` preserve approximate negative zero. Exact zero has no sign.

```cpp
#include <uni20/core/math.hpp>
auto small = mpreal("1e-100", p);
auto stable_exp = uni20::math::expm1(small); // Retains the small nonzero result.
auto stable_log = uni20::math::log1p(small);
auto exact_cube_root = uni20::math::cbrt(mpreal{-8}); // Exact -2.
auto finite_asinh = uni20::math::asinh(mpreal{1}, p);
```

Run `mpreal_elementary_example` for a comparison of naive and stable formulas
using the same generic code with `double` and `mpreal`.

The [coverage plan](scalar_math_coverage_plan.md) records the special-function
and nonstandard elementary extensions. This real expansion does not add corresponding MPC overloads.

Approximate real arithmetic follows MPFR's infinity/NaN behavior: real division by zero can
produce infinity or NaN, and `sqrt` of a negative value produces NaN. This differs
from exact rational division, whose result must remain rational. `isfinite`,
`isinf`, `isnan`, and `signbit` inspect real values, and ordering with NaN is
unordered. MPFR exponent bounds and exception flags retain the underlying
library's semantics; Uni20 does not change these settings.

`mpreal` models Uni20's `Real` concept and has a `uni20::numeric_limits`
specialization, so `has_numeric_limits_v<mpreal>` is true. Precision-dependent
queries require an exemplar or explicit precision: use
`numeric_limits<mpreal>::epsilon(x)` or `numeric_limits<mpreal>::epsilon(p)`,
and similarly `digits(x)` or `digits(p)`. The zero-argument `epsilon()` is
deleted; exact or unset exemplars cannot supply finite working precision.
The trait indicates that a specialization exists, not that type-only limits
are available. Existing generic algorithms may need further adaptation for
constants, allocation, and precision before accepting this scalar. See the
[runtime numeric-limits contract](scalar_policy.md#runtime-precision).

`mpreal` and `complex<mpreal>` own host-side GMP/MPFR/MPC state and cannot be
stored in `CudaBuffer`, `CudaTensor` or `CudaMatrix`. CUDA buffers and accessors
require trivially copyable elements; these owning scalars are rejected at
compile time. Convert explicitly to a supported native scalar before transfer.

## Arithmetic and numerical utilities

These real operations are available both as scalar overloads and through
`uni20::math`. Without an explicit precision, exact inputs retain exact results
where possible; a finite operand supplies the result precision. Multiple finite
numerical operands must match, except for sign sources and neighbor directions.
All operations reject unset operands.

| Family | Operations | Result |
| --- | --- | --- |
| Integer powers/roots | `pown(x,n)`, `rootn(x,n)` | Real scalar; negative orders mean reciprocal powers/roots |
| Integer rounding | `floor`, `ceil`, `trunc`, `round`, `round_even` | Real scalar, not an integer carrier; `round` breaks ties away, `round_even` to even |
| Fused arithmetic | `fma(a,b,c)` | `a*b+c`, with one final rounding, including mixed rational operands |
| Selection/sign | `fdim`, `fmin`, `fmax`, `copysign` | Positive difference, minimum, maximum, or magnitude with a new sign |
| Remainders | `fmod(a,b)`, `remainder(a,b)`, `remquo(a,b)` | Truncating or nearest-even quotient rule; `remquo` returns `{remainder, quotient}` |
| Decomposition | `frexp(x)`, `modf(x)`, `ilogb(x)` | `{fraction, exponent}`, `{fraction, integer}`, or a signed 64-bit exponent |
| Binary scaling | `ldexp(x,n)`, `scalbn(x,n)` | `x * 2^n`; these are identical for Uni20's binary formats |
| Paired evaluation | `sincos(x)`, `sinhcosh(x)` | `{sin, cos}` or `{sinh, cosh}`, owning components at the same precision |
| Adjacent values | `nextafter(x,direction)`, `next_up(x)`, `next_down(x)` | Adjacent value on the first operand's finite grid |
| Classification | `isfinite`, `isnan`, `isinf`, `signbit` | `bool`; exact rationals are finite, and exact zero has no negative sign |

The named result types live in `<uni20/core/math_results.hpp>`. Their real fields
own their values; no provider output pointers or borrowed components escape.

### Result precision and exact arithmetic

An optional trailing finite `Precision` on rounding, fused arithmetic,
selection/sign, remainders, decomposition, scaling and integer powers/roots
**rounds the result from the original operands**. It does not round inputs
first. This differs deliberately from the elementary-function input-conversion
contract above. For example:

```cpp
namespace m = uni20::math;
auto p = Precision::bits(3);
mpreal third("1/3", Precision::exact());
auto zero = m::fma(third, mpreal(3,p), mpreal(-1,p)); // Exactly zero, stored at p.
auto other = m::fma(third.at(p), mpreal(3,p), mpreal(-1,p)); // Nonzero.
auto integer = m::floor(mpreal("1023/1024", Precision::exact()), p); // Zero.
// Rounding the input to p first would produce one, and floor would then be one.
```

For `floor` and its relatives there are two distinct steps: determine the
mathematical integer, then represent that integer at the result precision using
nearest-even rounding. Thus `ceil(9217/1024, Precision::bits(2))` produces 8:
the mathematical ceiling is 10, which is halfway between 8 and 12 at two bits.
It is not a directed rounding of the original input onto the floating grid.
`modf(x,p)` determines both parts first and rounds each independently; its
rounded fraction can consequently reach magnitude one. `frexp(x,p)` instead
renormalizes a rounded fraction of magnitude one and increments the exponent.

`pown` and `rootn` accept ordinary integer types whose values fit signed 64 bits;
`bool` and floating orders are unsupported, and out-of-range unsigned values
throw `std::out_of_range`. Zero exponent gives one. A zeroth root is an error
for exact evaluation and NaN for finite evaluation. Exact powers and rational
perfect roots remain exact; other roots require precision. Even-order roots of
negative values have no real result and give NaN at finite precision. Zero to a
negative power/order gives infinity at finite precision and a domain error in
exact arithmetic.

MPFR integer powers and roots round once. Exact non-dyadic arguments use directed
bounds until both bounds select the same rounded result; rational perfect roots
and dyadic power results are handled separately to resolve exact ties. This
path requires the rational argument (after inversion for a negative power) to
lie within MPFR's configured exponent range; it throws `std::overflow_error`
otherwise. It never changes that range. Exact arithmetic may allocate large
numerators or denominators; requesting an exact enormous power or binary shift
is a request to construct that rational value.

`fmod` uses quotient truncation toward zero; `remainder` and `remquo` round the
quotient to the nearest integer, with ties to even. The latter's `quotient`
is the signed low **three** bits of that integer, in [-7,7], not the full
quotient. Remainders keep the dividend's sign when zero at finite precision.
Finite arithmetic with a zero divisor or nonfinite dividend gives NaN and zero
quotient bits; an infinite divisor returns the dividend. All-exact division by
zero throws. Mixed rational remainders and fused arithmetic keep large binary
exponents separate from their coefficients rather than expanding enormous
powers of two.

`frexp` uses `x = fraction * 2^exponent`, with a nonzero finite fraction of
magnitude in [1/2,1). Zero and nonfinite values are returned as the fraction with
exponent zero. `ilogb` returns the exponent of the leading binary digit
(`frexp(x).exponent - 1`) and throws `std::domain_error` for zero, infinity or
NaN. Unlike decomposition and scaling, exponent extraction has no precision
argument. Classification also has no precision argument.

`fmin`/`fmax` prefer the numeric operand to a NaN. If both inputs are zero,
`fmin` prefers negative zero and `fmax` positive zero. `fdim(a,b)` returns
positive zero when `a <= b`, otherwise `a-b`; a NaN input propagates.
`copysign` retains the magnitude's exactness/precision and takes only the sign
from its second argument. An exact zero cannot acquire a negative sign.

### Paired functions and adjacent values

`sincos` and `sinhcosh` follow their elementary components' input-conversion
contract: an explicit precision converts the input first. Approximate evaluation
uses MPFR's paired routine. Exact zero gives exact `{0,1}`; other exact inputs
need finite precision.

Adjacent values are different: exact rationals have no successor. The first
operand must be approximate, or a trailing finite `Precision` must select a
grid by rounding it first. The direction remains unrounded and may have any
precision. Equal zeros return the direction's sign. NaN propagates; stepping
outward from infinity leaves it infinite, while stepping inward gives the
largest finite value of that sign on the selected grid.

MPFR has no subnormals. The neighbors of zero are determined by MPFR's current
minimum exponent, and those of infinity by its maximum exponent. Native
floating types retain their native subnormals and exponent limits. Precision
alone sets the significand size, not an independent exponent range. Uni20 does
not change MPFR's exponent range or ambient rounding mode.

Run `mpreal_utilities_example` for result-rounding, exact decomposition and
mixed-precision direction examples.

## Complex scalars

```cpp
#include <uni20/core/mpcomplex.hpp>
auto p = uni20::Precision::bits(256);
uni20::complex<uni20::mpreal> z{"1.25", "-0.5", p};
auto product = z * uni20::conj(z); // Owning complex result.
auto magnitude = uni20::abs(z);   // Owning real at p.
auto component = z.real();        // Independent owning copy.
z.imag(uni20::mpreal{"0.1", uni20::Precision::bits(400)}); // Round to p.
```

Both components are exact, or both share one finite precision. Construction from
one exact and one approximate component uses the approximate component's
precision. Two approximate components must match unless a third `Precision`
argument requests conversion. `complex<mpreal>{}` and `complex<mpreal>{1}` are
exact zero and one. Explicit `uninitialized` constructs an unset placeholder.

Complex addition, subtraction, multiplication, division, conjugation, negation
and squared norm preserve exact rational components. `sqrt` returns exact
principal roots when both components are rational, and `abs` stays exact when
the magnitude is rational. Integer and half-integer powers use exact arithmetic;
positive real bases also support rational real powers. General complex
noninteger powers still require finite precision. Zero/unit identities for
`exp`, `log`, `sin`, `cos` and `arg(positive real)` remain exact. Exact zero has no
sign: on the negative real axis, the exact square root takes the upper cut side.
With an approximate operand, basic arithmetic retains exact components until
rounding each result component to its precision. Elementary functions continue
to use MPC's principal branches and signed imaginary zero after input conversion.

Complex `sqrt`, `exp`, `log`, `sin`, `cos`, `pow`, `abs`, `norm`, and `arg`
also accept a trailing finite `Precision`, with the same convert-then-evaluate
contract. `abs`, `norm`, and `arg` return an approximate `mpreal` at that precision.
MPC's principal branches apply after conversion; an approximate negative imaginary
zero retains its sign, whereas exact zero converts to positive zero.

Assignment adopts the complete source state. Component setters retain an
existing finite precision. On an exact complex value, an exact replacement stays
exact; an approximate replacement supplies precision for both components.
`real()` and `imag()` return independent owning values. Comparisons compare values
without rounding; NaN is unequal. No complex ordering is provided.

Borrowed MPC access requires an approximate value. `copy_to(mpc_ptr)` rounds
exact or approximate components into an initialized destination. Stream output is
`(real,imag)`; `format_scalar` uses the usual imaginary-unit suffix.

## Text and asynchronous use

Exact values print as canonical fractions (`1/3`) or integers (`1`). Formatting
options do not turn them into decimal approximations; convert with `.at(p)` first.
Both exact and finite-precision parsing accept fractions of signed base-ten
integers, such as `-2/3` or `4/-6`, and canonicalize them. Finite-precision parsing
rounds the rational once. Whitespace, decimal/exponent fraction components, and
multiple slashes are rejected; a zero denominator throws `std::domain_error`.
Canonical fraction output round-trips exactly through parsing at
`Precision::exact()`.

For approximate values, `value.to_string()` emits enough decimal digits to round-trip at the same
precision, without a native-float intermediate. `value.to_string(n)` requests
`n` significant digits, including a single digit when `n == 1`. Finite nonzero
output uses scientific notation; zero preserves its sign, and exceptional values
use `inf`, `-inf`, or `nan`. The string
constructor rejects incomplete input, whitespace, and embedded NUL. Stream
insertion uses the round-trip representation; stream precision does not override
it. `format_real`/`format_scalar` also support this type, including general, fixed,
and scientific notation and negative-zero normalization. Their default precision
is computed from the value. `parse_real<mpreal>(text, p)` requires explicit
precision; `read_real(stream, value)` uses the destination value's current
precision and leaves it unchanged on invalid input. Table schemas and runtime
precision selection are not yet connected.

Precision is carried by values and explicit `Precision` objects, including when
passed in coroutine frames. Scalar operations never read or change MPFR's default
precision or default rounding mode. They use explicit result precision and
rounding arguments; a suspended coroutine does not depend on a precision guard
remaining active. MPFR must have thread-local support because its exponent bounds
and exception flags are library state. Applications should not change exponent
bounds during Uni20 calculations.

`Async<mpreal>` can be initialized with an explicit scalar. Coroutine lambdas must
remain captureless and `static`, passing values or `Precision` as arguments.
The optional [MPLAPACK adapter](../linalg/mplapack_mpfr.md) establishes internal
temporary precision separately for every synchronous provider call after awaits.

## Tensor construction defaults

Host tensors accept a trailing `Precision`:

```cpp
auto p = uni20::Precision::bits(256);
uni20::DenseMatrix<uni20::mpreal> a(2, 3, p); // Actual zeros at 256 bits.
uni20::DenseMatrix<uni20::mpreal> work(uni20::uninitialized, 2, 3, p);
// work's scalar objects exist, but are unset until assigned.
auto view = uni20::reshape_view(a, 6);
a.default_precision(uni20::Precision::bits(80));
// Existing elements still have 256 bits; view.default_precision() now reports 80 bits.
auto rounded = uni20::at_precision(a, uni20::Precision::bits(80));
// rounded owns converted elements and has an 80-bit construction default.
```

The default belongs to the storage and survives copying, moving, empty shapes
and owning reshapes. It supplies the precision of new zeros during construction
or growth. Explicitly uninitialized allocation creates unset objects, without
skipping their C++ lifetimes. Ordinary nonempty zero construction without a
default throws; default-constructed empty tensors may acquire a default later.
`reset_shape(uninitialized, extents)` preserves an existing default or its absence;
allocating unset scalar objects does not require working precision.

Element assignment adopts that scalar's precision and does not update the
tensor default. Changing `default_precision(p)` changes metadata only; it never
scans or converts existing elements. `at_precision(tensor, p)` is the explicit
owning conversion and rejects unset source elements. A low-precision input's
lost digits cannot be recovered by conversion to a higher precision.

Parent-backed structural views inherit the parent's current default. Changing
an owner's default is visible through existing conjugation and reshape views,
including nested views; it does not convert their elements. Views do not offer
an independent default setter. Materializing an owning tensor copies the current
default, after which its metadata is independent of the source.

Async aliases may bind reserved storage before the parent is constructed. They
resolve precision through the parent only once the relevant epoch is readable,
just like other parent metadata. Reusing an alias across epochs observes each
readable parent state's default without a mutable cache. Raw mdspans carry no
tensor default and require explicit operation precision.

Writing through a structural view changes elements but does not change the
parent's construction default, even if the operation requests another working
precision. A future view that converts element precision would be a separate
numerical transformation, not an ordinary structural view.

`common_default_precision(a, b)` selects matching input defaults and throws if
either is missing or their finite defaults differ. Exact defaults are neutral;
two exact defaults produce `Precision::exact()`, which a numerical backend may
reject unless the caller supplies finite operation precision. It does not inspect individual elements.
`prepare_output(output, extents, p)` records the operation precision on owning
outputs and prepares unset storage when a new allocation is needed. Structural
view outputs keep their parent's default. Reused elements are not converted;
the numerical operation must overwrite the elements promised by its contract.

Run `mpreal_example` for exact fractions, an 80-digit calculation, pi generation,
and explicit mixed-precision conversion with explanatory output.

## Experiment scope and representation

The active payload is selected by a variant: unset, exact rational, or MPFR/MPC.
Approximate precision is queried from the native payload rather than duplicated.
The exact real payload reuses `exact_constant`; exact complex uses two rationals.
No ambient scalar precision or expression templates are introduced.

`mpreal_exact_example` demonstrates unchanged generic accumulation with `T{}`
and `T{1}`, exact complex arithmetic, and the explicit approximation boundary.
The experiment has not optimized payload size or rational allocation costs.
