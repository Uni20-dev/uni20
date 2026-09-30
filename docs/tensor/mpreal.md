# Arbitrary-precision real scalars

The first arbitrary-precision slice supplies `uni20::mpreal`, an owning MPFR
scalar, explicit `Precision`, exact rational literal arithmetic, and a
precision-aware pi constant. Optional MPC supplies `uni20::complex<mpreal>`.
It is a CPU scalar API. Tensor allocation, MPLAPACK kernels, and table/CLI precision
selection are subsequent integration work; enabling this option does not
advertise an arbitrary-precision BLAS or LAPACK backend.
`complex<mpreal>` and `make_complex_t<mpreal>` are unavailable without MPC, so
generic code cannot silently select the standard complex layout.

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

Default construction produces an unset `mpreal` without allocating MPFR storage.
Use `initialized()` to query this state. Copying, moving, swapping, destruction
and assignment are safe; assignment from a numerical value supplies its value
and precision. Numerical use, formatting, precision queries and native-handle
access on an unset object throw `std::logic_error`. Unset is neither zero nor NaN.
This permits allocate-then-assign buffers; it does not supply a zero accumulator.
Every new numerical value obtains precision
explicitly or from another scalar. `Precision::decimal_digits` uses a conservative
upper bound for conversion to binary bits and can allocate slightly more bits
than the mathematical minimum.

Copy and move construction and assignment preserve the source value and
precision. Moving leaves the source unset; self-move assignment preserves the
value. No arithmetic expression retains
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

Both components have one working precision. Construction from two `mpreal`
values requires matching precisions; a third `Precision` argument explicitly
converts both. Embedding one real value retains its precision and supplies
positive imaginary zero. Copy/move assignment adopts the whole source precision;
component setters preserve the existing complex precision. `real()` and `imag()`
return owning values, never references into MPC storage.

The complex type shares the real type's unset default state, owning value
semantics, nearest-even rounding and explicit `.at(p)` conversion. Arithmetic
requires matching precision; exact constants first round at the complex operand's
precision. Equality compares stored values without requiring equal precision,
and a NaN component makes equality false. No complex ordering is provided.

`conj`, `abs`, `norm`, `arg`, `sqrt`, `exp`, `log`, `sin`, `cos` and `pow` use
MPC. Square root and logarithm use the principal branch; signed imaginary zero
selects the side of the negative-real-axis cut. Classification uses the two
components: finite requires both finite; `isnan`/`isinf` report whether either
component has that classification. Round-trip component strings can reconstruct
the value with the two-string constructor. Stream output is `(real,imag)`;
`format_scalar` uses the existing presentation options and imaginary-unit suffix.

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
MPLAPACK's internal temporary precision requires separate integration; it is not
implied by Async scalar support.

## Tensor construction defaults

Host tensors accept a trailing `Precision`:

```cpp
auto p = uni20::Precision::bits(256);
uni20::DenseMatrix<uni20::mpreal> a(2, 3, p); // Actual zeros at 256 bits.
uni20::DenseMatrix<uni20::mpreal> work(uni20::uninitialized, 2, 3, p);
// work's scalar objects exist, but are unset until assigned.
auto view = uni20::reshape_view(a, 6);
a.default_precision(uni20::Precision::bits(80));
// Existing elements and view.default_precision() still have 256 bits.
auto rounded = uni20::at_precision(a, uni20::Precision::bits(80));
// rounded owns converted elements and has an 80-bit construction default.
```

The default belongs to the storage and survives copying, moving, empty shapes
and owning reshapes. It supplies the precision of new zeros during construction
or growth. Explicitly uninitialized allocation creates unset objects, without
skipping their C++ lifetimes. Ordinary nonempty zero construction without a
default throws; default-constructed empty tensors may acquire a default later.

Element assignment adopts that scalar's precision and does not update the
tensor default. Changing `default_precision(p)` changes metadata only; it never
scans or converts existing elements. `at_precision(tensor, p)` is the explicit
owning conversion and rejects unset source elements. A low-precision input's
lost digits cannot be recovered by conversion to a higher precision.

Views of existing tensors snapshot the default. Raw mdspans do not carry it.
Descriptors constructed from a pointer to reserved storage, including current
async alias factories, cannot inspect an unconstructed parent and start without
a default. They need an explicitly supplied default or operation precision.
Access to their parent must still respect its readable epoch. This avoids both
unsynchronized metadata reads and mutable caches inside read-only views.

`common_default_precision(a, b)` selects matching input defaults and throws if
either is missing or they differ. It does not inspect individual elements.
`prepare_output(output, extents, p)` records the operation precision and prepares
unset storage when a new allocation is needed. Reused elements are not converted;
the numerical operation must overwrite the elements promised by its contract.

Run `mpreal_example` for exact fractions, an 80-digit calculation, pi generation,
and explicit mixed-precision conversion with explanatory output.
