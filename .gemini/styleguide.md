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

For each finding, identify a reachable failure under the documented contract.
Check the language or build-tool semantics before proposing a fix.
