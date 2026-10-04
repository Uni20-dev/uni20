# Scalar math coverage plan

This plan covers the general-purpose real math surface for `mpreal` and the
matching `uni20::math` interface. It is broader than Bethe's immediate needs.
The implementation status below distinguishes the elementary-functions change
from follow-up work; it does not promise every listed function today.

The current contracts live in [scalar math dispatch](scalar_math_design.md),
[arbitrary-precision reals](mpreal.md) and [scalar policy](scalar_policy.md).
Complex MPC coverage is a separate extension: a real MPFR routine does not
establish a complex branch convention or implementation.

## Shared rules

- Add the scalar implementation, generic entry point, documentation and tests
  together. Generic algorithms use `uni20::math`, with the existing isolated
  ADL boundary. Do not add subsystem-specific lookup wrappers.
- Preserve exact rational results where the operation supports them. An
  all-exact operation requiring approximation needs explicit finite precision.
  Never choose ambient working precision or silently convert through `double`.
- Approximate inputs stay approximate, including mathematically integral
  results. Unary results retain input precision. Binary numerical operands
  normally require matching finite precision; exact operands are neutral.
- A trailing finite `Precision` on elementary functions converts the inputs
  first, then evaluates. This is the existing contract, not a promise to round
  the original rational expression only once. Basic arithmetic has its own
  stronger mixed-rational rounding contract; fused operations need an equally
  deliberate treatment.
- Document operation-specific exceptions to common precision, particularly
  sign sources, comparison operands and direction arguments. They do not all
  supply an approximation budget.
- Use provider routines where available. Stable elementary functions must use
  their dedicated routines, particularly `expm1` and `log1p`.
- Keep domain, NaN, infinity and signed-zero behavior explicit. Exact rational
  zero has no sign; finite-precision zero can have one. Unset values are errors.
- Native float32/64, supported float80 and float128 must retain their precision.
  A missing native or extension overload is unsupported, never a narrowing
  conversion. A nonstandard name needs a native implementation decision before
  it is advertised as precision-generic.
- Use eager owning values. No expression templates, implicit ambient precision,
  or new tensor/kernel dispatch rules are required for this scalar work.

## 1. Standard elementary functions

Status: implemented by the elementary-functions change, with the existing MPFR
4.1 minimum. This slice has no new MPFR-version requirement.

| Family | New `mpreal` overloads, with and without trailing `Precision` |
| --- | --- |
| Exponentials and logarithms | `exp2`, `expm1`, `log2`, `log10`, `log1p` |
| Roots | `cbrt` |
| Inverse trigonometry | `asin`, `acos` |
| Hyperbolic functions | `sinh`, `cosh`, `tanh`, `asinh`, `acosh`, `atanh` |

Expose all of these through `uni20::math`. Also expose existing scalar `tan`,
`atan`, `atan2` and `hypot` there. Existing `abs`, `sqrt`, `pow`, `exp`, `log`,
`sin` and `cos` remain part of the surface.

Exact behavior includes rational cube roots, rational powers of two and integer
base-two/base-ten logarithms of rational powers of the base, plus the zero/one
identities documented in `mpreal.md`. Other inputs need working precision.
Finite-precision evaluation uses MPFR round-to-nearest, ties-to-even on the
stored input, after any explicitly requested input conversion.

Validation includes cancellation at tiny positive and negative arguments,
precision retention across the shared scalar matrix, accuracy improvement at
higher precision, exact roots/powers/logarithms, explicit precision, exceptional
values and signed zero. MPFR-disabled native builds remain supported.

## 2. Arithmetic and numerical utility operations

Status: planned. Split into focused PRs rather than adding all signatures at
once. Settle each result and precision contract before implementation.

| Group | Remaining surface | Main design work |
| --- | --- | --- |
| Integer powers and roots | `pown`, `rootn` | Signed exponent/order types; zero and negative orders; rational perfect roots; native and float128 algorithms without narrowing |
| Rounding | `floor`, `ceil`, `trunc`, `round`, explicitly named ties-to-even rounding | Preserve exact rational semantics, ties and signed zero; avoid ambient rounding mode |
| Stable arithmetic | `fma`, `copysign`, `fdim`, `fmin`, `fmax` | Fused rounding with mixed rational inputs; sign/comparison operands; NaN rules |
| Remainders | `fmod`, `remainder`, `remquo` | Truncating versus nearest-even quotient; negative inputs and quotient-bit contract |
| Decomposition/scaling | `modf`, `frexp`, `ldexp`, `scalbn`, exponent extraction | Named result values; exponent width and special-value behavior |
| Paired functions | `sincos`, `sinhcosh` | Named owning results, consistent component precision, paired provider calls |
| Adjacent values | `nextafter`, next-up, next-down | A finite representable grid; direction need not have matching precision; signed-zero, infinity and exponent-limit rules |
| Classification | `isfinite`, `isnan`, `isinf`, `signbit` through `uni20::math` | Bool results need a dispatch policy separate from precision-preserving numerical results |

`ceil` and `ldexp` already exist in the generic facade for native types; MPFR
support remains to be added. Classification already exists for `mpreal` at the
root `uni20` namespace. Preserve those existing contracts when completing the
shared surface.

Prefer named results such as `{fraction, exponent}` or `{value, gamma_sign}`
to public output-pointer interfaces. Names are illustrative until each API is
reviewed. Exponent extraction must account for MPFR's exponent range instead
of automatically returning C's `int`. Distinguish a base-two decomposition
exponent from an `ilogb`-style exponent, including zero/nonfinite cases.

Exact rational operations must not detour through floating point: for example,
`floor(1/3)` is exact zero, and rational remainder is computed by an exact
quotient rule. `fma(a, b, c)` must not become `(a*b)+c`, including when one input
is a non-dyadic rational. Extend the existing mixed-arithmetic rounding tools
or use a proved sufficient/adaptive precision strategy with final rounding.

There is no next rational number. Adjacent-value APIs require a finite grid,
with explicit precision when it cannot be inferred. MPFR's in-place neighbor
functions are implementation primitives, not a ready-made C++ value API. Decide
whether `nextafter` follows the standard signed-zero convention and how native
subnormals differ from MPFR's representation before exposing it.

## 3. Real special functions and constants

Status: planned provider wrappers with documented domains. Start with small,
useful families and leave unavailable native operations constrained out until
an appropriate native provider is chosen.

| Family | Target surface |
| --- | --- |
| Gamma | `tgamma`, `lgamma`, signed log-Gamma, `digamma`, `beta`, upper incomplete Gamma |
| Error and zeta | `erf`, `erfc`, `zeta` |
| Exponential integral and dilogarithm | `Ei`, real-part `Li2` |
| Other provider functions | Integer-order Bessel `J_n` and `Y_n`, Airy `Ai`, `agm` |
| Constants | Explicit-precision pi (existing), log(2), Euler's constant, Catalan's constant |
| Optional convenience | Integer-argument factorial |

Use `lgamma` for `log(abs(Gamma(x)))`; expose the Gamma sign independently
rather than changing that meaning for negative Gamma. Name and document the
**upper** incomplete Gamma explicitly. A real-part dilogarithm is not a complex
dilogarithm API. Document provider limitations and behavior at domain boundaries;
wrapping a provider is not a guarantee over a wider domain.

Constants should follow the precision-aware descriptor approach already used by
pi. Keep symbolic constants separate from exact rationals, and require finite
precision for irrational values. Decide factorial's exact-rational behavior
alongside its integer-source and resource limits.

## 4. Nonstandard elementary extensions

Status: planned. These can be interleaved with stage 3 according to consumer
need. Some have been available in MPFR for years, others require newer versions.

- `exp10`, with an explicit native/float128 provider or documented algorithm.
- `sinpi`, `cospi`, `tanpi`, and pi-scaled inverse trigonometric functions.
- `log2p1`, `log10p1`, `exp2m1`, `exp10m1`, and `compound` for `(1+x)^n`.
- Lower priority: `sec`, `csc`, `cot`, `sech`, `csch`, `coth`.

Check each routine against the supported MPFR version. Keep capability checks
centralized in configuration, or make a deliberate minimum-version change;
do not introduce a header that unconditionally requires newer symbols while
CMake still advertises 4.1. Native support needs its own accurate implementation
or provider. Merely multiplying a rounded pi into the argument is not an
adequate replacement for a pi-scaled function's argument reduction contract.
Likewise, do not replace the small-argument functions with cancellation-prone
compositions.

## Verification and completion criteria

Each PR must include:

1. Scalar and facade availability tests, including rejection of conversion-only
   types and unsupported precision arguments.
2. Exact-state, explicit-precision and unset-state tests where applicable.
3. Numerical regressions in the shared [precision matrix](../development/numerical_testing.md):
   float32/64/80/128 and runtime MPFR at 128 and 256 bits, as configured.
   Add MPC cases only when complex support is actually implemented.
4. Discriminative checks for the failure being prevented: cancellation,
   narrowing, wrong rounding, wrong sign/domain, or loss of exactness. Provider
   agreement checks verify wrapper wiring; independent series, identities and
   analytic references establish additional numerical evidence.
5. Tests of precision-dependent error, retaining the tested type for comparison
   and using a wider reference where needed. No `EXPECT_NEAR` conversion to
   double for extended-precision claims.
6. GCC 13 and Clang 19 coverage, MPFR-enabled and disabled configurations, and
   an appropriate broader suite before publishing. Record tested provider
   versions; do not equate source compatibility with a test on that version.
7. Updated operation tables, domain/precision documentation and a useful example.

Track completed stages here, and keep current behavior in the canonical guides.
Deferred signatures and semantics remain proposals until their implementing PR.
The plan does not add scalar serialization, tensor-wide precision storage,
CUDA MPFR execution, or new linear-algebra providers.

## Provider references

- [MPFR 4.1 reference](https://www.mpfr.org/mpfr-4.1.0/mpfr.html): the current minimum.
- [MPFR transcendental functions](https://www.mpfr.org/mpfr-current/mpfr.html#Transcendental-Functions).
- [MPFR arithmetic functions](https://www.mpfr.org/mpfr-current/mpfr.html#Arithmetic-Functions).
- [MPFR integer and remainder functions](https://www.mpfr.org/mpfr-current/mpfr.html#Integer-and-Remainder-Related-Functions).
- [MPFR miscellaneous functions](https://www.mpfr.org/mpfr-current/mpfr.html#Miscellaneous-Functions).
