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

For each finding, identify a reachable failure under the documented contract.
Check the language or build-tool semantics before proposing a fix.
