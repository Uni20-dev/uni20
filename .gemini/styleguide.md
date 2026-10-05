# Uni20 Review Guidance

Follow [AGENTS.md](../AGENTS.md) and the
[review guide](../docs/development/code_review.md).

## Platform Scope

- Linux is the primary platform. Windows users should use the Linux build
  through WSL.
- macOS CPU portability is in scope, subject to Uni20's C++23 and
  compiler/library requirements. Do not infer tested support from a compiler
  name or version alone.
- Native Windows support is out of scope unless explicitly requested. Do not
  raise findings solely about MSVC, the Windows CRT, Win32 APIs, or native
  Windows build tooling. Report issues that also affect Linux or macOS.
- Existing Windows-specific branches, upstream dependency support, and
  synthetic Windows test fixtures do not establish native Windows support.

## Tensor Initialization

Ordinary owning Tensor shape constructors, including `DenseMatrix<T>(rows, cols)`,
zero-initialize numerical elements. Diagonal-only population is valid; untouched
elements remain zero. The trailing-precision form for `mpreal` and
`complex<mpreal>` creates zeros at that precision. To request uninitialized Tensor
shape construction, use the tag-first API, for example `DenseMatrix<T>(uninitialized, rows, cols)`.
Low-level raw-buffer allocation has a different contract.

Check the current [initialization contract](../docs/tensor/creation_and_reshape.md#owning-tensor-initialization)
before reporting uninitialized reads or suggesting a zero-fill. A previous
review's claim that DenseMatrix construction leaves garbage is not the current
API contract.

## Scalar Math and ADL

Use the current [scalar math contract](../docs/tensor/scalar_math_design.md)
when reviewing lookup and precision behavior. Distinguish the layers:

- Scalar-generic host algorithms call `uni20::math` from `core/math.hpp`.
  Class scalars, including `mpreal`, reach their scalar-specific overloads
  through its isolated ADL dispatch. The class lookup deliberately excludes
  standard floating overloads to reject conversion-only fallbacks; do not
  replace it mechanically with `using std::foo; foo(...)`.
- `core/detail/native_math.hpp` is the native adapter behind that dispatch.
  Its `NativeReal` constraint admits only `float`, `double`, `long double`,
  and configured `uni20::float128`. Qualified `std::` calls are appropriate
  here. Class scalars do not use this adapter.
- Resolve aliases before claiming a type needs ADL. In the supported MPLAPACK
  binary128 configurations, `mplapack_binary128_t` aliases `_Float128`,
  `__float128`, or `long double`; it is not a wrapper class. An alias does not
  give a fundamental type an associated namespace.
- An ADL or precision-loss finding must identify a concrete supported type,
  a call admitted by the constraints, and the failing lookup or narrowing.
  Check the provider overloads and relevant precision tests. Qualified calls
  alone are not evidence of a defect; a missing native provider overload
  should be reported with the affected configuration and call.

Apply older review advice about unqualified math only where it fits the
current lookup boundary. The shared dispatcher and its native adapters are
intentional; they are not subsystem-private replacements for the math API.

## Finding Evidence and Duplication

For each finding, identify a reachable failure under the documented contract.
Check the language or build-tool semantics before proposing a fix.
Report one finding per shared root cause and list affected sites together.
Separate comments are useful when they demonstrate distinct failures or need
different fixes, not for each occurrence of the same disputed idiom.
