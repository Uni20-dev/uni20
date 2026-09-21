# Using Uni20 from another CMake project

Uni20 can be consumed directly from source using `add_subdirectory` or CMake
FetchContent. It currently does not provide an installed CMake package or
exported targets for `find_package(uni20)`.
Source integration requires CMake 3.28 or newer, including for CUDA builds.

## Local checkout

```cmake
cmake_minimum_required(VERSION 3.28)
project(my_solver LANGUAGES CXX)

set(UNI20_SOURCE_DIR "" CACHE PATH "Path to a Uni20 source checkout")
add_subdirectory("${UNI20_SOURCE_DIR}" "${CMAKE_CURRENT_BINARY_DIR}/uni20")

add_executable(my_solver main.cpp)
target_link_libraries(my_solver PRIVATE uni20_core uni20_common)
```

Configure with `-DUNI20_SOURCE_DIR=/path/to/uni20`. Set
`-DUNI20_ENABLE_MPLAPACK=ON` to enable Uni20's binary128 scalar and backend;
see [MPLAPACK setup](../linalg/mplapack_binary128.md) for provider selection.
Use the configured Uni20 targets so generated headers, compile definitions,
and provider libraries travel together.

## Pinned FetchContent dependency

```cmake
include(FetchContent)
FetchContent_Declare(uni20
  GIT_REPOSITORY https://github.com/Uni20-dev/uni20.git
  GIT_TAG <tested-full-commit-hash>
)
FetchContent_MakeAvailable(uni20)
target_link_libraries(my_solver PRIVATE uni20_core uni20_common)
```

Replace the placeholder with a tested commit. For development against a sibling
checkout, configure with
`-DFETCHCONTENT_SOURCE_DIR_UNI20=/path/to/uni20`. CMake uses those existing
sources without downloading or updating them, and compiles them inside the
consumer's build tree. This does not reuse an existing Uni20 binary build.
If a parent already supplies Uni20, reuse its targets rather than adding a
second copy with a different configuration.

## Targets and settings

- `uni20_core` provides scalar foundations, configured types, and build metadata.
- `uni20_common` provides shared diagnostics, presentation, and terminal helpers.
- Other components, such as `uni20_linalg`, may be linked as needed.
- `uni20` is the aggregate library target.

The targets advertise the C++23 requirement transitively. A consumer need not
set a global language standard to use them. If consumer public headers expose
Uni20 facilities, use `PUBLIC` linkage (or `INTERFACE` for a header-only library).
Set `CMAKE_POSITION_INDEPENDENT_CODE=ON` before adding Uni20 when its static
libraries will be linked into a shared library or Python extension.

Source integration still configures Uni20's full library and numerical
dependencies, including oneTBB and BLAS/LAPACK. Linking a component limits what
that target requires; it is not a component-only configure mode.

When embedded, `UNI20_BUILD_TESTS`, `UNI20_BUILD_COMBINED_TESTS`,
`UNI20_BUILD_EXAMPLES`, `UNI20_BUILD_BENCH`, `UNI20_BUILD_ASM`,
`UNI20_BUILD_PYTHON`, `UNI20_BUILD_DOCS`, and `UNI20_BUILD_FORMAT_TARGET` default
to `OFF`. Each can be enabled explicitly. Standalone defaults remain `ON`.
These are initial cache defaults: changing how a checkout is consumed does not
reset an existing build cache. Use a separate build directory for each setup.

`UNI20_ENABLE_LTO` also defaults to `OFF` when embedded and `ON` standalone.
IPO/LTO does not propagate through target linkage. A parent opting into Uni20
LTO must also enable compatible IPO on its final executables or shared libraries;
otherwise, for example, Clang's bitcode archives can fail to link. The parent can
instead configure IPO for the whole build using CMake's
`CMAKE_INTERPROCEDURAL_OPTIMIZATION[_<CONFIG>]` variables before creating targets.

Uni20 disables C++ module scanning in its own directory scope because its
targets use headers rather than named modules. The parent's
`CMAKE_CXX_SCAN_FOR_MODULES` setting is preserved.

When enabling `UNI20_BUILD_TESTS`, the parent must call `enable_testing()` (or
include `CTest`) at its source root for `ctest --test-dir <parent-build>` to
discover the embedded tests. Enabling tests only in Uni20's subdirectory does
not enable root-level discovery.

Uni20 leaves the parent's build type, configuration list, compiler flags, and
BLAS selection hints intact. Uni20-specific dependency configuration stays in
its directory scope. Optional dependency settings, such as `BUILD_GMOCK` and
`TBB_TEST`, use Uni20 defaults only when the parent has not set them. Required
provider settings still apply when Uni20 populates that provider. FetchContent
populates a shared dependency once, so parents should declare their dependency
choices before adding Uni20.

Uni20's FetchContent declarations mark dependency headers as `SYSTEM` and use
`EXCLUDE_FROM_ALL` unless `UNI20_BUILD_EXTERNAL_TESTS=ON`. Linked dependency
targets still build when needed; unrelated dependency targets and install rules
are excluded from the enclosing project's default build/install. Explicitly
requested dependency components remain available as named targets. A parent's
earlier FetchContent declaration takes precedence over these declaration defaults.
Warning options for compiling dependency sources are controlled separately by
`UNI20_EXTERNAL_NO_WARN` and `UNI20_EXTERNAL_NO_WERROR`.

If the parent has already created `BLAS::BLAS` or `LAPACK::LAPACK`, it must set
`BLA_SIZEOF_INTEGER` to the actual integer ABI of those targets before adding
Uni20: `4` with `UNI20_ILP64=OFF`, or `8` with `UNI20_ILP64=ON`. The declaration
must describe all supplied BLAS/LAPACK targets. Uni20 rejects missing, `ANY`,
or conflicting declarations; CMake targets have no standard ABI metadata, so
Uni20 trusts the parent's explicit declaration rather than probing the binary.
This requirement also applies with `UNI20_BACKEND_BLAS=OFF`, since LAPACK still
uses BLAS. When neither target exists, Uni20 selects its own providers using
`UNI20_ILP64` and leaves the parent's hints unchanged.

Supplied targets carry their own libraries and link options; Uni20 does not
add legacy `BLAS_LINKER_FLAGS` or `LAPACK_LINKER_FLAGS` to them. Supplying both
targets avoids BLAS/LAPACK package discovery entirely, without requiring
`BLAS_FOUND`, `LAPACK_FOUND`, or a separately discoverable installation. Missing
providers still require discovery, including their transitive dependencies.
BLAS vendor detection uses the supplied target's direct link information and
imported library locations, including import libraries. Configuration-specific
locations follow the active build configurations, `MAP_IMPORTED_CONFIG_<CONFIG>`,
and `IMPORTED_CONFIGURATIONS`. With CMake 4.2 or newer, detection uses CMake's
resolved location to honor the `CMP0200` policy recorded on the supplied target,
which may differ from Uni20's policy scope. After configuration selection, a missing
runtime or import-library property falls back independently to its unsuffixed
property. An artifact shared by several configurations can leave the companion
artifact ambiguous; conflicting candidate vendors report
generic BLAS, since vendor macros and extension availability apply to every
build configuration.
Opaque targets default to generic BLAS unless `UNI20_BLAS_VENDOR` explicitly
identifies the provider for vendor extensions.

Uni20 defaults to storing dependency sources and builds under
its own binary directory when embedded; standalone builds retain the shared
source-cache default. `UNI20_FETCHCONTENT_BASE_DIR`,
`UNI20_FETCHCONTENT_SOURCE`, and `UNI20_FETCHCONTENT_SOURCE_BASE_DIR` remain
available as explicit overrides.

Explicitly enabled developer targets are named `uni20_doc`,
`uni20_clang_format`, and (with GCC coverage enabled) `uni20_coverage` when
embedded. Standalone builds retain `doc`, `clang_format`, and `coverage`.
Their inputs and outputs refer to Uni20's own source and binary directories.
`UNI20_DOCS_WEB=ON` always enables the documentation target.
