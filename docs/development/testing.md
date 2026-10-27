# Testing uni20

This document describes how uni20 tests are configured, built, and run.

## Overview

uni20 testing uses:

- [Google Test](https://github.com/google/googletest) for C++ unit tests
- [CTest](https://cmake.org/cmake/help/latest/manual/ctest.1.html) for test discovery and execution
- CMake [`gtest_discover_tests`](https://cmake.org/cmake/help/latest/module/GoogleTest.html) for automatic registration

Tests live in `tests/` and are organized by module.

Scalar-generic numerical changes use the shared cases and evidence requirements
in [Numerical precision validation](numerical_testing.md). A type compiling or
an aggregate suite passing does not certify every operation at that precision.

Register tests for configured features and applicable supported combinations.
Keep known coverage gaps in the numerical coverage report, and fail when
expected support is missing. Reserve runtime skips for conditions such as
unavailable hardware topology; a fully configured build should otherwise run
all its registered tests.

## Integration checks before pushing

Changes to shared scalar traits, numeric limits, initialization, or other widely
included templates need complete builds and unfiltered CTest runs in the affected
CI configurations. Building only the new numerical tests can miss failures in
storage, formatting, CLI examples, and Python bindings that consume those same
headers. Keep examples, CLI, and bindings enabled wherever the corresponding CI
job enables them. A successful targeted run is useful during development but is
not a replacement for this integration check.

For cross-cutting scalar changes, cover the GCC Debug and Release configurations,
Clang Debug, and the optional MPLAPACK configuration when affected. Include the
arbitrary-precision family disabled and enabled, with and without MPLAPACK
kernels. The Clang job disables the optional float80 alias so that native
`long double` support is tested independently of that alias. Run an
unfiltered `cmake --build <build-dir>` before `ctest --test-dir <build-dir>
--output-on-failure`; test discovery after a partial build is not evidence that
the entire suite built. Record compiler/configuration details, actual failures,
and skip counts, and identify unavailable configurations explicitly. Do not claim
GPU, MPI, or macOS validation from a Linux CPU run.

## Sanitizer boundaries

Use `-DUNI20_SANITIZE=address,undefined` to instrument Uni20. Commas and
semicolons are both accepted, including mixed lists. Quote semicolon-separated
arguments in a shell, for example `'-DUNI20_SANITIZE=address;undefined'`.
Compiler flags, dependency policies, and sanitizer test registration all use
the same normalized list.

An embedding application that supplies global `-fsanitize` flags must also set
this option to activate Uni20's dependency policy; the option alone does not
instrument parent-directory application targets.

The compiled oneTBB scheduling runtime (`TBB::tbb`) is excluded from address,
undefined-behavior and thread instrumentation because of known provider
incompatibilities. In particular, the instrumented resumable-task teardown in
oneTBB 2022.3 binds a reference through a null `thread_data` pointer. This boundary
predates that reproducer and does not claim that oneTBB has passed sanitizer validation.
The exclusion is private to the actual build target: Uni20, application code
and inline oneTBB code compiled in those consumers retain their checks.
Imported oneTBB libraries are left unchanged; their instrumentation is determined
by whoever built them.
For ASan/TSan builds, the fetched provider keeps its thread-backed resumable-task
implementation even without instrumentation, avoiding a switch to
`makecontext`/`swapcontext`, which ASan does not fully support.

`UNI20_DEBUG_DAG` deliberately retains `NodeInfo` metadata for process-lifetime
diagnostics. This is an accepted diagnostic allocation policy, not ownership of
the calculation's result payload. When leak-checking with DAG diagnostics
enabled, or MPFR caches on terminating provider threads, explicitly select
Uni20's test suppression policy:

```sh
DEBUGINFOD_URLS= \
ASAN_OPTIONS=detect_leaks=1:halt_on_error=1 \
LSAN_OPTIONS=suppressions=/path/to/uni20/tests/sanitizers/lsan.supp:print_suppressions=0 \
UBSAN_OPTIONS=halt_on_error=1:print_stacktrace=1 \
  ctest --test-dir build --output-on-failure
```

Automated sanitizer runs should clear `DEBUGINFOD_URLS` in their test environment
so symbolization does not depend on remote debug-symbol downloads. A stalled
request can otherwise consume the test timeout while a sanitizer formats its
report. This preserves local symbolization and every sanitizer check; system
library stack traces may have less detail when their debug symbols are not
installed locally. Keep this policy in test commands or CI/preset environments,
not in the library or the user's global shell configuration. See
[LLVM's debuginfod options](https://llvm.org/docs/CommandGuide/llvm-symbolizer.html#cmdoption-llvm-symbolizer-debuginfod).

The file matches allocation stacks in `uni20/async/async_node.hpp`, the
debug-metadata header. Matching the source path also covers inlined factories
whose qualified names are absent from reduced debug information. Leak detection
remains enabled; general Async allocation stacks are not suppressed.

The same file suppresses observed MPFR cache/pool allocations through the exact
C functions `mpfr_atan`, `mpfr_cos`, and `mpfr_cache`. These caches belong to
operating-system threads and can remain allocated after a provider thread
terminates. They are separate from the result storage owned by `mpreal`, which
is released by its destructor. See [MPFR's memory-handling contract](https://www.mpfr.org/mpfr-current/mpfr.html#Memory-Handling).
This is an opt-in test policy, not runtime cleanup or a claim that every MPFR
allocation is harmless. A function-based rule also excludes other internal
allocations with that function in their stack; do not broaden it to GMP,
`libmpfr`, or all `mpreal` allocations.

Keep the `^...$` anchors: LSan matches patterns against every symbolized stack
frame, and an unanchored `mpfr_atan` also matches the C++ wrapper
`mpreal_access::apply<&mpfr_atan>`. That wrapper allocates the owning result
*before* calling MPFR, so a substring rule could hide a real leaked payload.
Linux ASan/LSan builds with MPFR register `Sanitizers.MpfrCachePolicy`, which
compares retained caches with explicit local cleanup and verifies that leaked
`mpreal` payloads and ordinary heap allocations still fail. The isolated probe
disables LSan's TLS roots to expose cache allocations without depending on
worker scheduling or termination; ordinary application tests retain TLS roots.
Local symbolization must be available for these checks; set
`ASAN_SYMBOLIZER_PATH` to the matching `llvm-symbolizer` if it is not on `PATH`.
See [LLVM's suppression documentation](https://clang.llvm.org/docs/LeakSanitizer.html#suppressions).

Downstream applications should select this file from their pinned Uni20 source
instead of maintaining independent MPFR rules. Suppression statistics are quiet
so that successful CLI calls keep their usual stderr behavior; actual errors
still print and fail. This does not
change allocation or lifetime behavior in ordinary builds. For a complementary
leak check without retained diagnostic pointers, use `UNI20_DEBUG_DAG=OFF`.

Runtime reclamation is deferred. The async documentation records future options
for [diagnostic metadata](../async/task_registry_debug.md#diagnostic-memory-lifetime-and-deferred-reclamation-options)
and [MPFR caches](../async/tbb_execution_primer.md#mpfr-cache-lifetime-and-deferred-cleanup-options).
These proposals add no cleanup API.

## Test Layout

Current test modules include:

- `tests/async`
- `tests/backend`
- `tests/cli` (only when `UNI20_BUILD_CLI=ON`)
- `tests/common`
- `tests/cmake` (downstream configure, build, and execution regressions)
- `tests/core`
- `tests/docs` (Python documentation-checker regressions, run by Docs PR CI)
- `tests/kernel`
- `tests/krylov`
- `tests/linalg`
- `tests/mdspan`
- `tests/models`
- `tests/numerics` (shared real/complex precision and backend matrix)
- `tests/sanitizers` (optional instrumentation and suppression probes)
- `tests/symmetry`
- `tests/tensor`
- `tests/python` (only when `UNI20_BUILD_PYTHON=ON`)

`tests/krylov` contains Uni20-native C++ matrix-free interface tests, native
solver tests, Matrix Market fixture checks, and optional MPLAPACK binary128
probes when `UNI20_ENABLE_FLOAT128=ON` and `UNI20_BACKEND_MPLAPACK=ON`.

## Configuration Options

Primary CMake options:

The `ON` defaults below apply to standalone builds. Tests, combined tests, and
Python bindings default to `OFF` when Uni20 is embedded in another project.

| Option | Default | Description |
|--------|---------|-------------|
| `UNI20_BUILD_TESTS` | `ON` | Build and register C++ tests |
| `UNI20_BUILD_COMBINED_TESTS` | `ON` | Build `tests/uni20_tests` combined test executable |
| `UNI20_BUILD_PYTHON` | `ON` | Also enable `tests/python` CTest entries |
| `UNI20_ENABLE_FLOAT80` | `AUTO` | Detect native 80-bit scalars; `ON` requires them, `OFF` omits them |
| `UNI20_ENABLE_FLOAT128` | `OFF` | Binary128 scalar family (`AUTO/ON/OFF`) |
| `UNI20_ENABLE_MPREAL` | `OFF` | Arbitrary-precision real and complex scalar family (`AUTO/ON/OFF`) |
| `UNI20_BACKEND_MPLAPACK` | `OFF` | Enable MPLAPACK kernels for all enabled optional scalar families |

Example configure:

```bash
cmake -S . -B build -DUNI20_BUILD_TESTS=ON -DUNI20_BUILD_COMBINED_TESTS=ON
```

To disable tests:

```bash
cmake -S . -B build -DUNI20_BUILD_TESTS=OFF
```

To enable the optional MPLAPACK binary128 probes, configure MPLAPACK separately
and point Uni20 at the resulting CMake package. See
[MPLAPACK Binary128 Setup](../linalg/mplapack_binary128.md) for the full package
build and validation workflow:

```bash
cmake -S . -B build \
  -DUNI20_ENABLE_FLOAT128=ON -DUNI20_BACKEND_MPLAPACK=ON \
  -Dmplapack_DIR=/path/to/mplapack-build-binary128
```

## Building Tests

After configuration, build with:

```bash
cmake --build build -j
```

## Running Tests

List discovered tests:

```bash
ctest --test-dir build -N
```

Run all tests:

```bash
ctest --test-dir build --output-on-failure
```

Run only matching tests:

```bash
ctest --test-dir build --output-on-failure -R IterationPlan
```

You can also run binaries directly:

```bash
./build/tests/common/uni20_common_tests
./build/tests/uni20_tests --gtest_filter=TraitsTest.*
```

## Separate vs Combined Test Executables

uni20 supports both modes:

- Per-module executables like `uni20_common_tests`, `uni20_async_tests`
- Combined executable `uni20_tests`

When both modes are enabled, CTest registers the per-module executables discovered via `gtest_discover_tests(...)`. The combined `uni20_tests` binary is built for manual runs, but it is not registered with CTest by default.

## Initialization diagnostics

Owning tensors initialize stored numerical values to zero by default.
The explicit `uni20::uninitialized` tag requests values that must be written
before use; see [tensor construction](../tensor/creation_and_reshape.md#owning-tensor-initialization).
Diagnostic filling and memory-tool annotations apply to fresh uninitialized
allocations, including internal scratch buffers. Acquiring mutable access to an
existing tensor does not change its initialization state.

| Setting or tool | Behavior |
| --- | --- |
| `UNI20_FILL_UNINITIALIZED_SNAN=AUTO` (default) | Fill supported floating-point values, including both complex components, with signaling NaNs in `Debug` and `DebugOpt` |
| `UNI20_FILL_UNINITIALIZED_SNAN=ON` / `OFF` | Enable or disable that fill in every configuration, independently of `NDEBUG` |
| Clang `UNI20_SANITIZE=memory` | Track undefined host values with MemorySanitizer; annotations follow the active translation unit's instrumentation |
| `UNI20_ENABLE_VALGRIND=ON` (default `OFF`) | Compile Memcheck annotations; requires `valgrind/memcheck.h`, and works with GCC or Clang |
| GCC/Clang address, undefined-behavior, or thread sanitizers | Complement these checks; they do not track general uninitialized numerical values |

Signaling NaNs help numerical failures become visible but do not guarantee a
trap. Uni20 does not change the process's floating-point exception settings.
Unsupported scalar types receive no invented NaN representation. Nontrivial
objects retain their required construction and destruction; diagnostic poisoning
does not invalidate their bookkeeping.

Host memory-tool annotations mark fresh values undefined **after** the optional
sNaN fill. Consequently, writing a diagnostic bit pattern does not conceal a
missing write from MSan or Memcheck. Instrumented assignments define the bytes
they write, and copies preserve the source's definedness. These tools detect
use of undefined values, not redundant zero initialization before a complete
overwrite.

For a GCC or Clang Memcheck build, for example:

```bash
cmake -S . -B build_codex/memcheck -DCMAKE_BUILD_TYPE=Debug \
  -DUNI20_ENABLE_VALGRIND=ON
cmake --build build_codex/memcheck --target uni20_common_tests uni20_initialization_diagnostic_probe
ctest --test-dir build_codex/memcheck --output-on-failure -R '^InitializationDiagnostic\.'
valgrind --error-exitcode=97 build_codex/memcheck/tests/common/uni20_common_tests \
  --gtest_filter='Initialization.*'
```

The subprocess tests distinguish complete initialization, a missing write, and
copying an undefined value. The latter two succeed as tests only when the tool
reports the expected error. They register automatically when the corresponding
tool configuration is enabled; Memcheck tests also require the `valgrind`
executable. Run MSan and Memcheck configurations separately.

For Clang MSan, set `UNI20_SANITIZE=memory` in a separate build. Origin tracking
can be enabled with `-fsanitize-memory-track-origins=2`. MSan requires compatible
instrumentation of dependencies or precise annotations at external boundaries;
an ordinary uninstrumented BLAS/LAPACK or C++ library is not a fully supported
MSan execution environment. This initial integration verifies the allocation
and copy hooks with small subprocess tests. It does not yet annotate all
BLAS/LAPACK or device-transfer boundaries. See the
[MSan requirements](https://clang.llvm.org/docs/MemorySanitizer.html#handling-external-code)
and [Memcheck client requests](https://valgrind.org/docs/manual/mc-manual.html#mc-manual.clientreqs).

For CUDA, run Compute Sanitizer's `initcheck` on a build configured with
`UNI20_FILL_UNINITIALIZED_SNAN=OFF`. A GPU diagnostic fill is itself a write,
so leaving it enabled would hide missing initialization from `initcheck`.
Host MSan and Memcheck annotations do not track GPU loads and stores. See
[CUDA buffer initialization](../backends/cuda/buffers.md) and the
[initcheck documentation](https://docs.nvidia.com/compute-sanitizer/ComputeSanitizer/index.html#initcheck-tool).

The `CMakeSubproject.initialization_*` tests check configuration propagation, including
Ninja Multi-Config, explicit overrides, and deliberate disagreement with
`NDEBUG`.

## Adding New Tests

### CMake integration tests

CI runs the GCC Debug build with CMake 3.28.3 and the Clang Debug build with a
current CMake 4.x release. This covers the supported minimum and newer imported
target policies explicitly, independently of the runner's preinstalled tools.

Run `ctest --test-dir build --output-on-failure -R '^CMakeSubproject\.'` to
configure, build, and execute small parent projects using `add_subdirectory`
and FetchContent. The consumers deliberately request C++17, own common helper
target names, and check that Uni20 preserves their settings. Their linked
Uni20 targets must supply C++23, generated headers, and transitive libraries.
The parent enables C++ module scanning; Uni20 disables it only for its own
targets and preserves the parent's setting.
Embedded LTO defaults to off, so the parent can link Uni20 without enabling IPO
on its own targets. The optimized Ninja Multi-Config consumer checks this with
the outer build's compiler, including Clang's bitcode-sensitive archive linkage.
Small IPO fixtures build and run a static library and its executable with
automatic IPO enabled, disabled, unsupported, or inherited from a parent. They
check that Debug skips the probe, that directory defaults do not alter earlier
dependencies or parent targets, and that all four optimized configurations work
with Ninja Multi-Config.
Additional checks cover parent-owned package hints and shared dependency
options. ABI contract checks accept matching 4/8-byte declarations and reject
missing, ambiguous, invalid, and conflicting declarations for existing BLAS and
LAPACK targets. An embedded consumer reuses real parent LP64 targets, calls BLAS
and LAPACK, and verifies that the provider targets remain unchanged; another
checks ABI rejection before Uni20 changes its local discovery hints.
Synthetic imported targets additionally check reuse with package discovery
disabled, both with and without the BLAS backend. They deliberately supply
unrelated package variables to detect accidental reuse of stale link flags or
vendor metadata. A dependency export containing multiple include directories
inside a generator expression checks that warning handling preserves valid
install exports, as required by fetched MPLAPACK.
A small fetched dependency also checks an actual default build: linked libraries
must build and run, and a warning-producing dependency header must compile in
the consumer under `-Werror`. An unrelated dependency target is excluded by
default and included with `UNI20_BUILD_EXTERNAL_TESTS=ON`.
A dependency-argument check requires a specific diagnostic for omitted or empty
`TARGET` arguments before the helper can perform package discovery.
Vendor-detection checks cover imported configuration mappings, fallback
locations, and import libraries, using CMake's resolved locations as an
independent oracle. The configuration-only vendor fixture models a Windows
target system to exercise independent DLL/import-library fallbacks on every
host without requiring a Windows compiler or runtime. On CMake 4.2 or newer,
both `CMP0200` modes are tested with the detector defined under the opposite
policy, ensuring that the imported target's saved policy governs resolution.
A separate package-hint check resolves private packages in
a fresh child configure through the same arguments used by the consumers.
On Unix with a Python interpreter, recording stand-ins execute the
documentation and formatting target commands and validate their arguments
without rewriting the source checkout. When Python bindings are enabled in the
outer build, an embedded consumer builds the extension and runs its Python
tests from the parent build root. It reuses installed GoogleTest and nanobind
packages or fetched sources according to the outer build's dependency policy;
when GoogleTest is fetched, it also checks requested GoogleMock population.
When the outer build uses an
installed MPLAPACK package, the consumers also compile and execute binary128
arithmetic and a BLAS call. Fetched MPLAPACK builds use ordinary precision in
these nested checks to avoid repeatedly compiling the full provider;
the main test suite covers their binary128 operations.

These tests reuse the outer build's dependency source locations or package
paths and generator platform, toolset, and instance settings, with separate
build directories under `tests/cmake`. The consumer retains `ConsumerSpecial`
and generates the standard configurations, `DebugOpt`, and any requested custom
configuration. Ninja builds additionally exercise a real Ninja Multi-Config
consumer in `RelWithDebInfo`. The Python package hints and disconnected-fetch
policy are forwarded so an offline outer build does not silently require
downloads in its nested consumers. These tests register
directly with CTest because they test CMake integration rather than C++ units.

### C++ unit tests

1. Add the source file in the appropriate `tests/<module>/` directory.
2. Add it to that module’s `tests/<module>/CMakeLists.txt` via `add_test_module(...)`.

Example:

```cmake
add_test_module(mdspan
  SOURCES
    test_concepts.cpp
    test_iteration_plan.cpp
    test_transform_view.cpp
    test_new_case.cpp
  LIBS
    uni20_mdspan
)
```

## Numerical Comparisons

Choose the comparison from the numerical contract being tested:

| Intent | Preferred check |
| --- | --- |
| Exact integer, enum, shape, pointer, or state equality | `EXPECT_EQ` / `ASSERT_EQ` |
| Direct floating or complex scalar result, where a small ULP difference is the expected rounding uncertainty | `EXPECT_FLOATING_EQ` / `ASSERT_FLOATING_EQ` from `<uni20/common/gtest.hpp>` |
| Residual, reconstruction error, orthogonality defect, or cancellation result expected near zero | An explicit absolute or scale-aware error bound |
| Iterative algorithm, eigensolver, matrix function, or transcendental result | A tolerance derived from precision, problem scale, and algorithmic error |

`EXPECT_FLOATING_EQ` supports Uni20's native float32/64/80/128 real types,
the corresponding `uni20::complex<T>` types, and optional `mpreal`.
MPFR operands follow the [runtime-precision assertion contract](numerical_testing.md#choosing-a-numerical-assertion);
`complex<mpreal>` is not yet supported. Prefer it for
scalar-generic or complex tests, and over separate real/imaginary `EXPECT_NEAR`
checks or casting an extension scalar to `double`. GoogleTest's
`EXPECT_FLOAT_EQ` and `EXPECT_DOUBLE_EQ` have similar ULP semantics for tests
that are intentionally fixed to those built-in types.

Do not use ULP distance mechanically:

- Comparing a cancellation residual with exact zero can report a very large ULP
  distance even when the residual is numerically excellent.
- Reconstruction and orthogonality checks should measure a norm or component
  error against a scale-aware tolerance.
- Iterative and decomposition algorithms can accumulate more than a fixed
  number of ULPs while still satisfying their mathematical accuracy contract.

See [Trace Macros](../diagnostics/trace_macros.md#googletest-integration) for
the assertion syntax and supported scalar types.

## Internals

`cmake/Uni20TestHelpers.cmake` defines `add_test_module(...)`, creates per-module test executables, and accumulates sources/libs for the optional combined executable.

## Linux crash handler slowdown (Apport)

uni20 includes many GoogleTest death tests (`EXPECT_DEATH`/`ASSERT_DEATH`) that intentionally call `std::abort()` to verify contract failures.

The configuration of some Linux systems may interpret these as system crashes which are directed to Apport, via the `kernel.core_pattern` sysctl knob. When this is enabled, each death test crash can launch an Apport Python helper. With many death tests, this can dramatically slow down the total test time.

Check current setting:

```bash
cat /proc/sys/kernel/core_pattern
```

For development and CI environments focused on fast death-test runs, prefer:

```bash
sudo sysctl -w kernel.core_pattern=core
```

To persist this setting you could use

```bash
echo "kernel.core_pattern=core" | sudo tee /etc/sysctl.d/99-local-core-pattern.conf
sudo sysctl --system
```
or disable the `apport` service.
