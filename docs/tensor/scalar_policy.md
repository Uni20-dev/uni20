# Scalar Type Policy

Optional [arbitrary-precision real scalars](mpreal.md) use `uni20::mpreal`
with explicit runtime `Precision` when `UNI20_ENABLE_MPFR=ON`. This first
layer provides scalar arithmetic and exact constants. `UNI20_ENABLE_MPC=ON`
additionally supplies MPC-backed `complex<mpreal>` and `make_complex_t<mpreal>`;
these names are unavailable in a real-only build. Default-constructed MPFR/MPC
scalars are exact zero; explicit `uninitialized` supplies an unset placeholder.
They hold exact rationals until finite working precision is supplied. Their
precision cannot be described by type-only numerical limits. Their `Real`/`Complex` traits do not imply BLAS/LAPACK support. Tensor
allocation and dense backend integration are separate from scalar support.
`UNI20_ENABLE_MPLAPACK_MPFR=ON` adds a narrowly scoped
[matrix-product and LU-solve backend](../linalg/mplapack_mpfr.md), without extending
the type-only `LapackScalar` concepts or generic Krylov support.

`uni20::complex<T>` selects the appropriate owning scalar type. Existing native
real types retain exact `std::complex<T>` type identity. Generic complex APIs
should deduce the complex type directly (`template <Complex C>`) and obtain its
real type through `make_real_t<C>`; deduction through the selecting alias is not
supported, including in partial specializations or nested container parameters.
Tests should exercise deduction without explicit template arguments or optional
real-valued arguments that could mask this limitation. Internal adaptations
specifically for the standard-library family use `detail::standard_complex<T>`,
including its fixed-precision LAPACK ABI and CUDA storage/execution adapters.
MPC component access returns owning real values; setters convert to the existing
complex precision. Do not reinterpret an MPC value as adjacent C++ real objects.

Runtime-precision host tensors take a trailing `Precision` and carry a construction
default independently of their elements. Parent-backed structural views inherit
that default, resolving it within the readable parent epoch for async aliases;
changing a default never converts stored values. Use `at_precision(tensor, p)` for
explicit bulk conversion. Fixed-precision storage and views have no corresponding
runtime state. See [tensor construction defaults](mpreal.md#tensor-construction-defaults)
for output allocation and view metadata semantics.

This page records the project scalar spelling and concept policy. The concrete
aliases live in `src/uni20/core/types.hpp`; scalar traits and concepts live in
`src/uni20/core/scalar_traits.hpp` and `src/uni20/core/scalar_concepts.hpp`.

## Scalar Spelling

Uni20 code should spell real and complex scalar types through the project-level
aliases when an alias exists.

The canonical real aliases are:

| alias | meaning | availability |
| --- | --- | --- |
| `uni20::float32` | `float` | always |
| `uni20::float64` | `double` | always |
| `uni20::float80` | native x87 extended precision, alias of `long double` | only when `UNI20_HAS_FLOAT80=1` |
| `uni20::float128` | configured binary128 real scalar | only when `UNI20_HAS_FLOAT128=1` |

`uni20::float80` has 64 significand bits (about 19 decimal digits) and the x87
extended exponent range. CMake automatically detects the native `long double`
format from its numerical limits; it does not infer precision from `sizeof`.
An fp80 object can occupy 12 or 16 bytes because of padding. No provider,
software emulation, or silent substitution is added: on platforms where
`long double` is binary64 or binary128, `UNI20_HAS_FLOAT80=0` and the alias is
absent. Ordinary `long double` remains a valid Uni20 real scalar on all those
platforms. Guard direct references to `uni20::float80` with `UNI20_HAS_FLOAT80`.

`uni20::float128` is a configuration-dependent type. In the current MPLAPACK
configuration it aliases `mplapack_binary128_t`, whose concrete spelling is
selected by the resolved MPLAPACK package. Code that is not gated by
`UNI20_HAS_FLOAT128` must not name `uni20::float128`.

Runtime-facing code should use `uni20::ScalarPrecision` and
`uni20::visit_scalar_precision` from `uni20/core/scalar_precision.hpp` instead
of repeating configuration guards. The precision enum always recognizes
`fp32`, `fp64`, `fp80`, and `fp128`; the visitor throws when a recognized precision is
not configured. The visitor is the central preprocessor boundary that maps a
runtime precision to `uni20::float32`, `uni20::float64`, or the conditional
`uni20::float80` and `uni20::float128` types. `uni20::configured_scalar_precisions()` and
`uni20::configured_scalar_precision_choices()` expose the available set for
help text, diagnostics, examples, and future language bindings.

To enable fp128, configure with `UNI20_ENABLE_MPLAPACK=ON`. CMake prefers a
compatible installed MPLAPACK 3.0.0 or newer package and otherwise fetches the
pinned 3.0.0 release with only its binary128 backend enabled. See [MPLAPACK
Binary128 Setup](../linalg/mplapack_binary128.md) for dependency selection,
optional system-package, and validation commands.

For native real types, `uni20::complex<T>` retains standard-library ABI, layout
and interoperability. The MPC specialization has its own representation and
must be explicitly converted at provider boundaries.

Owning Tensor shape construction initializes stored numerical elements to zero.
`uni20::uninitialized` explicitly requests storage whose values must be supplied
before use. Copies and materializations initialize from their source, and
internal overwrite outputs request uninitialized allocation to avoid an extra
zero-fill. The low-level `HostBuffer(size)` retains its allocation-only numerical
contract; initialization-aware storage factories receive a `StorageInitialization`
choice from the tensor layer.

The public `uni20::enable_uninitialized_storage<T>` variable template in
`common/initialization.hpp` governs allocation and lifetime mechanics, not the
Tensor default. Its conservative default accepts trivially copyable and
trivially destructible types. Uni20 explicitly opts `uni20::complex<Real>` into
this behavior because the project relies on its scalar-array representation and
trivial lifetime behavior on supported standard libraries. Extension value
types may specialize this trait only when allocation, copying object
representations, and release without constructors/destructors form a valid
lifetime model. Other types retain required object construction and destruction.
Zero construction requires a numerical zero or a valid `T{}` value; requesting
it for a type without either value is rejected. Raw uninitialized allocation
remains available for lifetime-safe types without a default constructor.

Optional diagnostic filling uses `uni20::numeric_limits<Real>::signaling_NaN()`
when `has_signaling_NaN` is true, including both components of complex values.
Unsupported scalar types are not given an invented sentinel. MemorySanitizer
and Memcheck annotations can still track lifetime-safe uninitialized storage
without a NaN representation. Diagnostic filling and initialization tracking
are independent of `NDEBUG`; see [testing](../development/testing.md#initialization-diagnostics).

The canonical complex aliases are:

| alias | meaning |
| --- | --- |
| `uni20::complex<T>` | project-level complex scalar alias |
| `uni20::complex64`, `uni20::cfloat` | `uni20::complex<float>` |
| `uni20::complex128`, `uni20::cdouble` | `uni20::complex<double>` |
| `uni20::complex256`, `uni20::cfloat128` | `uni20::complex<uni20::float128>` when `UNI20_HAS_FLOAT128=1` |

Project code, tests, examples, and documentation should use
`uni20::complex<T>` unless they are explicitly documenting or testing the alias
relationship to `std::complex<T>`, or they are at a narrow external interop
boundary that must name the standard-library type.

CUDA execution does not change the logical or persistent scalar type.
`CudaTensor<uni20::complex<T>>` stores the same `uni20::complex<T>` objects as
the corresponding host tensor. CUDA accessors load and store their guaranteed
two real components through `cuda::std::complex<T>` execution values and
assignable proxy references. `cuComplex` and `cuDoubleComplex` remain localized
provider ABI types at cuBLAS call boundaries. `uni20::logical_value_t<T>` maps a
domain-specific execution value back to the backend-neutral logical scalar
advertised by expression accessors.

## Scalar Concepts

Scalar-generic code should prefer the scalar traits in `uni20/core`:

| trait/concept | use |
| --- | --- |
| `uni20::Real<T>` | real scalar constraints |
| `uni20::Complex<T>` | complex scalar constraints |
| `uni20::RealOrComplex<T>` | real-or-complex scalar constraints |
| `uni20::ScalarValued<T>` | scalar, container, or view whose recursive `value_type` resolves to a scalar |
| `uni20::RealScalarValued<T>` | scalar-valued type whose extracted scalar is real |
| `uni20::ComplexScalarValued<T>` | scalar-valued type whose extracted scalar is complex |
| `uni20::IntegerScalarValued<T>` | scalar-valued type whose extracted scalar is integer |
| `uni20::RealOrComplexScalarValued<T>` | scalar-valued type whose extracted scalar is real or complex |
| `uni20::BlasReal<T>` | real scalar with a configured dense BLAS-style backend |
| `uni20::BlasComplex<T>` | complex scalar with a configured dense BLAS-style backend |
| `uni20::LapackReal<T>` | real scalar with a configured dense LAPACK-style backend |
| `uni20::LapackComplex<T>` | complex scalar with a configured dense LAPACK-style backend |
| `uni20::LapackRealOrComplex<T>` | real or complex scalar whose underlying real precision has dense real LAPACK coverage |
| `uni20::LapackComplexReal<T>` | real precision whose `uni20::complex<T>` has dense complex LAPACK coverage |
| `uni20::make_real_t<T>` | underlying real scalar |
| `uni20::make_complex_t<T>` | complexified scalar/container type |
| `uni20::scalar_t<T>` | scalar extracted from a container-like type |
| `uni20::numeric_limits<T>` | project-level numeric limits customization point |
| `uni20::isfinite(x)` | project-level finite-value predicate for integer, real, and complex scalars |

`Scalar`, `Real`, `Complex`, `Integer`, and `RealOrComplex` constrain the type
itself. Use them for scalar-only helpers such as `uni20::conj`, `uni20::herm`,
or scalar arithmetic kernels. The `ScalarValued` family follows `scalar_t<T>`
through recursive `value_type` definitions, so it may match containers, views,
mdspans, or future tensor types. Use scalar-valued concepts when an algorithm
is generic over an object that carries scalar elements, not when the parameter
must itself be the scalar value.

`Blas*` and `Lapack*` describe configured backend support. This distinction
matters for extension scalar types: a type can be a valid Uni20 real scalar
without having BLAS or LAPACK coverage in the current build.

In particular, `float80` and `complex<float80>` do **not** acquire BLAS, LAPACK,
or GPU provider coverage. They reuse the existing native `long double` scalar
and generic CPU paths where the operation supports them. Runtime precision
visitors instantiate every configured scalar: clients requiring LAPACK must
guard those instantiations with the relevant capability concept and reject
unsupported requests, rather than silently narrowing. Native fp80 availability
does not extend the supported precisions of the projected LAPACK-based Krylov
solvers or DMRG.

Reference sums and inner products use compensated accumulation in the input
scalar field. Norms use scaled sum-of-squares arithmetic in the associated real
field. Algorithms must choose their numerical accumulation method explicitly;
Uni20 does not define a universal "next wider" scalar because no such type
exists for the widest configured precision, and silent promotion would make
result and performance behavior depend on the input dtype. Integer sums are not
accepted until Uni20 defines their overflow contract.

The CPU dense matrix exponential likewise performs norm estimation and
scaling-exponent arithmetic in the matrix scalar's real field. Its overflow
protection comes from logarithmic entry bounds and prescaling before high-order
matrix powers, so binary128 support does not depend on a nonexistent wider
floating-point type.

For matrix-free algorithms, avoid duplicating scalar type information in
interfaces when it can be inferred from the vector operations. For example, the
return type of `inner_product(x, y)` names the vector scalar field, and `norm(x)`
returns the associated real scalar. Additional scalar declarations should only
be added when they encode information that cannot be inferred from the
operation interface.

Future higher-precision real types should be integrated by extending the Uni20
real scalar traits, `uni20::numeric_limits<T>`, and the required linear algebra
backends. `uni20::BlasReal<T>`, `uni20::BlasComplex<T>`,
`uni20::LapackReal<T>`, and `uni20::LapackComplex<T>` are intentionally
backend-relative and separately extensible: they mean Uni20 has configured
dense BLAS-style or LAPACK-style implementations for that scalar, not that the
type is one of the standard Fortran BLAS/LAPACK ABI scalar types. A type may
eventually satisfy only one of these concepts if Uni20 has only one backend
layer for it. For example, an MPLAPACK-enabled build may make
`mplapack_binary128_t` satisfy both `BlasReal` and `LapackReal`, while
`uni20::complex<mplapack_binary128_t>` satisfies the corresponding complex
concepts only for paths with explicit complex MPBLAS/MPLAPACK wrappers. The
BLAS side can be project overloads backed by MPBLAS entry points rather than
`s`/`d`/`c`/`z` Fortran ABI symbols.

Krylov algorithms that form dense projected Hermitian problems use
`LapackRealOrComplex`: a complex vector field can still reduce to a real
projected tridiagonal problem. Algorithms that form dense complex projected
problems use `LapackComplexReal` or `LapackComplex`, because real LAPACK support
for the underlying precision is not sufficient there.

`uni20::make_complex_t<T>` follows the Uni20 real scalar traits rather than
`std::floating_point<T>`, so extension real types can opt into the scalar model
without depending on standard-library concept recognition. If the platform
supports `std::complex<Real>` for such a real type, `uni20::complex<Real>` should
continue to be the preferred spelling. If a future backend requires a different
complex representation, that change should happen behind the project-level
alias/traits boundary rather than by scattering backend-specific complex types
through algorithms.

## Scalar Math

Scalar-generic code should use `uni20::isfinite(x)` instead of directly calling
`std::isfinite(x)` when `x` may be a Uni20 scalar. Integers are always finite,
real scalars are checked for NaN and positive/negative infinity through
`uni20::numeric_limits<T>`, and complex scalars are finite only when both
components are finite. This keeps extension scalar support behind the same
project customization points as scalar spelling and numeric limits.

Use the typed constants in `std::numbers`, such as
`std::numbers::pi_v<Real>`, when a mathematical constant participates in
scalar-generic arithmetic. Do not widen an untyped `double` constant or a
decimal `double` literal into a higher-precision scalar. The supported GNU
binary128 configuration provides full-precision typed constants for
`uni20::float128`; its tests verify that the result retains precision beyond a
widened `double`. Uni20 does not currently duplicate `std::numbers`. If a future
scalar provider cannot supply suitable typed standard constants, introduce a
project customization point when integrating that provider.

## Floating-Point Comparisons

`uni20/common/floating_eq.hpp` provides ULP comparisons for binary32,
binary64, configured binary128, and native fp80, including their
`uni20::complex<T>` counterparts. `IeeeBinaryReal` still denotes only the
supported IEEE interchange layouts; `UlpOrderedReal` also admits native fp80.
The fp80 ordering uses exact exponent/significand decomposition, not object
bytes, so storage padding and byte order do not affect the distance.

`uni20/common/gtest.hpp` exposes this through `EXPECT_FLOATING_EQ` and
`ASSERT_FLOATING_EQ`. The optional third argument is a nonnegative ULP bound
(default four). Signed zeros compare equal; NaNs never do; infinities match
only when equal. The comparison uses the full distance even when the signed
diagnostic distance saturates at `std::numeric_limits<long long>::max()`.
The tracing `CHECK_FLOATING_EQ` and `PRECONDITION_FLOATING_EQ` assertions
share the same comparison implementation.

Use ULP bounds for contracts expressed in representable-value steps.
Algorithmic residuals, truncation errors, and reference data with limited
accuracy generally need explicit absolute/relative tolerances in the native
scalar type instead. Do not narrow fp80 or fp128 through GoogleTest's
double-based `EXPECT_NEAR`.

## Scalar Formatting

Scalar-generic diagnostics and presentation code should use
`uni20::format_scalar` or the corresponding presentation helpers rather than
assuming standard stream or formatter support. Trace formatting recognizes all
types satisfying Uni20's `Real` and `Complex` concepts, including
`uni20::float128` and `uni20::complex<uni20::float128>` when configured.

Native `float80` uses the existing `long double` parsing, formatting, and
presentation precision policy; no conversion through `double` is involved.

Scalar formatting support does not imply typed table or metadata support.
`presentation::DataTableValue` admits supported fixed-precision real types;
`mpreal` and optional `mpreal` are rejected by table schemas and `metadata_value`
construction/conversion until runtime-precision parsing and export contracts are
defined. `format_real(mpreal)` remains available for explicit textual output.
See [typed data tables](../diagnostics/data_tables.md) for the supported types.

Trace precision is independently configurable for float32, float64, and
float128 values. The global environment variables are
`UNI20_FP_PRECISION_FLOAT32`, `UNI20_FP_PRECISION_FLOAT64`, and
`UNI20_FP_PRECISION_FLOAT128`; append `_MODULE_<MODULE>` for a module-specific
override. Complex values use the precision of their real component type.

## Numeric Limits

Scalar-generic Uni20 algorithms should use `uni20::numeric_limits<T>` rather
than naming `std::numeric_limits<T>` directly. The primary Uni20 template is
undefined: there are no generic default values. A constrained specialization
explicitly delegates native arithmetic types to their specialized standard-library
limits. Every library scalar must supply its own Uni20 specialization; a standard
specialization alone does not opt a library type into the Uni20 interface.

Half-integer limits are defined by `common/half_int.hpp`:
`numeric_limits<basic_half_int<T>>` describes an exact, bounded, signed scalar
with radix 2 and the value-bit count of its doubled integer representation.
`is_integer` is false because values may have a fractional half. `lowest()` and
`max()` return exact half-integer endpoints. Infinity and NaN capabilities are
false; epsilon, floating-point exponent bounds, and an ambiguous `min()` query
are not provided. `uni20::is_exact(half_int_value)` consequently returns true.

Generic code must establish the numeric category before querying its limits.
An ordinary `&&` expression does not prevent template instantiation of its
right-hand side. Use nested `if constexpr` branches or constraints, and probe
optional quantities with `requires`. In particular, storage initialization must
not instantiate numeric limits for strings or other nonnumeric element types.

For extension or library scalar types, specialize `uni20::numeric_limits<T>`:

```cpp
namespace uni20
{
template <> struct numeric_limits<my_real>
{
    static constexpr bool is_specialized = true;
    static constexpr int digits = /* ... */;

    static my_real epsilon();
    static my_real epsilon(my_real const&); // Same value, for generic callers.
    static my_real min();
    static my_real max();
};
} // namespace uni20
```

Do not add specializations of `std::numeric_limits` for compiler fundamental
extension types such as `__float128` or `_Float128`. Those are not
user-defined types. Keep such support behind the Uni20 customization point.

### Runtime precision

`numeric_limits<mpreal>` separates fixed type properties (`radix`, `is_exact`,
`is_integer`, etc.) from quantities requiring a working precision:

```cpp
auto p = Precision::bits(256);
mpreal x("0.1", p);
auto eps = numeric_limits<mpreal>::epsilon(x); // 2^(1 - 256), at 256 bits
auto bits = numeric_limits<mpreal>::digits(x);
auto same_eps = numeric_limits<mpreal>::epsilon(p);
```

Scalar-generic code can use `numeric_limits<Real>::epsilon(value)` for native
real types as well. Their existing zero-argument, constexpr queries remain
available. Runtime `digits(value)` and `digits(p)` apply to `mpreal`; native
`digits` remains the usual compile-time constant.

An exact or unset exemplar cannot supply finite working precision: these runtime
queries throw rather than return zero or choose an ambient default. Pass a finite
`Precision` explicitly when starting an approximate algorithm from exact inputs.
`numeric_limits<mpreal>::epsilon()` is deleted, and other unsupported limits are
absent. `has_numeric_limits_v` indicates a specialization exists, not that every
query is available without a value. Generic algorithms requiring type-wide limits
must constrain those particular queries. This prevents the unspecialized standard
template's zero-valued fallback from silently becoming a convergence tolerance.

### Exactness queries

`uni20::is_exact(value)` queries the active arithmetic state of `mpreal`,
`complex<mpreal>` and `exact_constant`, and the static exactness of ordinary scalar
types. It returns false for an unset runtime scalar. `numeric_limits<T>::is_exact`
remains a compile-time type property; it cannot describe the changing state of an
individual `mpreal`. Ordinary integers are exact, while native floating values
are approximate even when numerically integral. Exact rational roots, powers,
parsing and explicit native conversions are described in [mpreal](mpreal.md).

Mixed exact/approximate basic arithmetic retains exact operands until rounding
the result to the finite operand's precision. Finite `+`, `-`, `*` and `/`
results are correctly rounded (per component for complex values); division
requires a nonzero divisor for this statement. Mixed complex multiplication and
division, and exact-real/approximate-real division, keep large binary exponents
separate from rational coefficients. The large-exponent path uses exact scaled
comparisons to select the rounded result, including cancellation and ties;
temporary integer storage scales with significand and rational sizes rather than
the numerical exponent. Ordinary-sized exponents use direct rational arithmetic. Two approximate operands still
require matching precision. Approximation-driven algorithms must obtain finite
working precision explicitly or from their inputs; an all-exact state supplies
no approximation budget.

Runtime-precision real and complex math functions accept a trailing finite
`Precision`: they convert each input to that precision, then evaluate, always
returning an approximate result at that precision. Without an override, exact
results remain exact where supported, and approximate operands supply the working
precision. This distinction also applies to the real outputs of complex `abs`,
`norm`, and `arg`.

## Numerical validation

Scalar support is validated per operation and backend using the
[shared numerical precision cases](../development/numerical_testing.md).
Float80 and finite MPFR/MPC values participate alongside float32/64/128; MPFR
runs at both 128 and 256 bits. Unsupported algorithms and missing optional
providers are reported explicitly. Passing scalar arithmetic or one backend's
tests does not establish library-wide support for that scalar.
