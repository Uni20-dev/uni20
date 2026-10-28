# Scalar families and numerical backends

Choose arithmetic independently of linear-algebra providers:

| Option | Default | Public types |
| --- | --- | --- |
| `UNI20_ENABLE_FLOAT80` | `AUTO` | `float80`, `complex160` |
| `UNI20_ENABLE_FLOAT128` | `OFF` | `float128`, `complex256` |
| `UNI20_ENABLE_MPREAL` | `OFF` | `mpreal`, `mpcomplex` (also `complex<mpreal>`) |
| `UNI20_BACKEND_MPLAPACK` | `OFF` | MPLAPACK kernels for enabled families |

`float32`, `float64`, `complex64`, and `complex128` are unconditional.
Complex format names count both components: `complex128` contains two
binary64 components, not two binary128 components.

## Requests and results

All three scalar options accept:

- `ON`: require usable support; missing dependencies or incompatible formats
  are configuration errors.
- `OFF`: omit the optional family and its provider kernels.
- `AUTO`: detect usable support without downloading a scalar provider;
  unavailable support leaves the family disabled.

For binary128, `AUTO` does not enable a family whose dependency policy forces
fetching (`UNI20_USE_SYSTEM_MPLAPACK=OFF`). If combined mpreal/binary128
conversions are unavailable, an explicit `ON` request takes priority over
`AUTO`; with both families automatic, mpreal takes priority.

`float80` requires native x87 extended-format `long double`: 64 significand
bits and the corresponding exponent range. Storage may include padding. It is
not emulated on other platforms. Disabling the alias does not disable the C++
builtin type.

The arbitrary-precision family requires installed GMP, MPFR >= 4.2, and MPC.
Real and complex scalars are enabled together. Working precision remains an
explicit per-value or per-operation choice, not a CMake setting.

Binary128 currently obtains its scalar type, math, and usage requirements from
MPLAPACK >= 3.0.0. Thus `UNI20_ENABLE_FLOAT128=ON` can fetch that dependency
even with `UNI20_BACKEND_MPLAPACK=OFF`. This does not register Uni20's MPLAPACK
kernels: its binary128 BLAS/LAPACK traits remain disabled. Extracting a
standalone scalar provider is a possible future implementation improvement.

Requests are not capability macros. C++ code tests the generated 0/1 values
`UNI20_HAS_FLOAT80`, `UNI20_HAS_FLOAT128`, and `UNI20_HAS_MPREAL`.
Provider integration tests use `UNI20_HAS_MPLAPACK_FLOAT80`,
`UNI20_HAS_MPLAPACK_FLOAT128`, and `UNI20_HAS_MPLAPACK_MPREAL`.
These are derived results, not independent configuration inputs.
The configure summary and build information report arithmetic and kernel
providers separately.

## Selecting MPLAPACK

`UNI20_BACKEND_MPLAPACK=ON` requests kernels for all enabled optional families.
There are no per-precision backend switches or separate MPC switch. Missing
required package components fail configuration instead of silently dropping
requested kernels. Requesting the backend with no enabled family also fails.
Binary128 GEMM also uses the existing `UNI20_BACKEND_BLAS` wrapper switch
(on by default); disabling that switch leaves its LAPACK kernels available.
MPFR and binary80 adapters do not have this additional GEMM requirement.

```sh
# Scalars, exact rationals and tensor storage; no MPLAPACK kernels.
cmake -S . -B build_codex/scalars -DUNI20_ENABLE_MPREAL=ON

# Binary128 and arbitrary precision, plus native float80 where available.
cmake -S . -B build_codex/wide \
  -DUNI20_ENABLE_FLOAT128=ON -DUNI20_ENABLE_MPREAL=ON \
  -DUNI20_BACKEND_MPLAPACK=ON

# Only arbitrary-precision kernels, without a binary128 compiler requirement.
cmake -S . -B build_codex/arbitrary \
  -DUNI20_ENABLE_FLOAT80=OFF -DUNI20_ENABLE_FLOAT128=OFF \
  -DUNI20_ENABLE_MPREAL=ON -DUNI20_BACKEND_MPLAPACK=ON
```

Dependency sourcing remains independent: `UNI20_USE_SYSTEM_MPLAPACK=AUTO`
prefers an installed package and otherwise fetches the pinned release; `ON`
requires an installed package; `OFF` fetches. Explicit backend selection may
fetch its components even if a scalar was auto-detected.

Scalar availability does not promise every numerical operation supports that
type. See [binary80](mplapack_binary80.md), [binary128](mplapack_binary128.md),
and [arbitrary-precision coverage](mplapack_mpfr.md). For example, binary80
complex LU and decompositions remain disabled for providers with unsafe generic
complex division.

## Updating older build trees

The old `UNI20_ENABLE_MPFR`, `UNI20_ENABLE_MPC`, `UNI20_ENABLE_MPLAPACK`,
`UNI20_ENABLE_MPLAPACK_BINARY80`, and `UNI20_ENABLE_MPLAPACK_MPFR` options are
removed, not silently interpreted. Remove those cache entries or start a fresh
build tree. In particular, the old `UNI20_ENABLE_MPLAPACK=ON` meant binary128;
its replacement is `-DUNI20_ENABLE_FLOAT128=ON -DUNI20_BACKEND_MPLAPACK=ON`.
