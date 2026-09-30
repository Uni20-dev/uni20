# Arbitrary-precision real scalars

The first arbitrary-precision slice supplies `uni20::mpreal`, an owning MPFR
scalar, explicit `Precision`, exact rational literal arithmetic, and a
precision-aware pi constant. It is a CPU scalar API. Tensor allocation,
MPC-backed `uni20::complex<mpreal>`, MPLAPACK kernels, and table/CLI precision
selection are subsequent integration work; enabling this option does not
advertise an arbitrary-precision BLAS or LAPACK backend.
`complex<mpreal>` and `make_complex_t<mpreal>` are deliberately unavailable in
this slice, so generic code cannot silently select the standard complex layout.

## Configuration

Install development packages, then enable the optional scalar layer:

```sh
sudo apt install libgmp-dev libmpfr-dev
cmake -S . -B build_codex/mpfr -DUNI20_ENABLE_MPFR=ON
cmake --build build_codex/mpfr --target uni20_mpfr_tests mpreal_example
ctest --test-dir build_codex/mpfr --output-on-failure -R 'MpReal'
```

MPFR 4.1 or newer and a thread-safe MPFR build are required. There is no
source-download fallback. `CMAKE_PREFIX_PATH` supports a non-system prefix;
individual paths can be supplied through `UNI20_GMP_INCLUDE_DIR`,
`UNI20_GMP_LIBRARY`, `UNI20_MPFR_INCLUDE_DIR`, and `UNI20_MPFR_LIBRARY`.
CMake checks compilation and linking, and checks MPFR thread-local support
when target executables can run. Scalar construction also checks thread-local
support, including in cross builds. Consumers linking `uni20_core` inherit the
required includes and libraries. With the option off, those dependencies are
not searched for or linked.

MPC is not needed for this real-only slice. Install `libmpc-dev` for the planned
complex layer. The MPFR scalar option is independent of
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

`mpreal` has no default constructor. Every new numerical value obtains precision
explicitly or from another scalar. `Precision::decimal_digits` uses a conservative
upper bound for conversion to binary bits and can allocate slightly more bits
than the mathematical minimum.

Copy and move construction and assignment preserve the source value and
precision. After move construction, the source is zero at its original precision;
move assignment also leaves its source valid. No arithmetic expression retains
references to its operands.

Binary `mpreal` arithmetic, including compound assignment, requires equal
precision. A mismatch throws `std::invalid_argument` before modifying either
operand. Explicit conversion uses `x.at(p)` or `mpreal{x, p}`; reducing precision
rounds, and increasing precision cannot restore digits already lost.
Compound arithmetic updates the destination in place, including when both
operands are the same object, without constructing a separate result scalar.

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
and `long double` conversions are available.

## Exact literal arithmetic

```cpp
constexpr auto decimal = 0.12345678901234567890123456789_mp;
auto exact = 0.1_mp + 0.2_mp;      // Owning exact_constant equal to 3/10.
auto third = 1_mp / 3_mp;          // Owning exact_constant equal to 1/3.
auto a = exact.at(p);             // Round once to p.
auto b = x + third;               // Round third to x's precision, then add.
```

The `_mp` literal records source characters without conversion through a machine
float. Its descriptor can be `constexpr`; operations on descriptors materialize
owning GMP rationals at runtime. `exact_constant` accepts decimal text for runtime
input and formats as a reduced numerator/denominator, or an integer.

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

Mixing an exact constant with `mpreal` first rounds the constant at the real
operand's precision, then performs the real operation. Parentheses therefore
identify a rounding boundary:

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

## Constants, elementary functions, and exceptional values

```cpp
auto pi_value = pi<mpreal>.at(p);
auto twice_pi = mpreal(2, p) * pi<mpreal>; // Infer p, evaluate immediately.
auto root = sqrt(mpreal(2, p));
auto spacing = epsilon(p);                // 2^(1-p.bit_count()).
```

The pi descriptor computes pi at the requested precision, unlike promoting a
previously rounded approximation. It supports arithmetic with an existing
`mpreal`; precisionless combinations involving pi are unsupported. This slice
supplies `pi<mpreal>`, not a generic replacement for `std::numbers`.

The scalar math functions include `abs`, `sqrt`, `exp`, `log`, `sin`, `cos`,
`tan`, `atan`, `atan2`, `hypot`, and `pow`. Results retain their operand precision;
binary functions require matching precision. Functions such as `sqrt` do not
accept precisionless constants in this slice.

Real arithmetic follows MPFR's infinity/NaN behavior: real division by zero can
produce infinity or NaN, and `sqrt` of a negative value produces NaN. This differs
from exact rational division, whose result must remain rational. `isfinite`,
`isinf`, `isnan`, and `signbit` inspect real values, and ordering with NaN is
unordered. MPFR exponent bounds and exception flags retain the underlying
library's semantics; Uni20 does not change these settings.

`mpreal` models Uni20's `Real` concept, but type-only numerical limits cannot
supply its runtime precision. `has_numeric_limits_v<mpreal>` is false; algorithms
must use explicit/value-derived precision and `epsilon(p)` rather than
`numeric_limits<mpreal>::epsilon()`. Existing generic algorithms may need further
adaptation for constants, allocation, and precision before accepting this scalar.

## Text and asynchronous use

`value.to_string()` emits enough decimal digits to round-trip at the same
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
Tensor storage factories and MPLAPACK's internal temporary precision require
separate integration; these are not implied by Async scalar support.

Run `mpreal_example` for exact fractions, an 80-digit calculation, pi generation,
and explicit mixed-precision conversion with explanatory output.
