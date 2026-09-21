# Using Uni20 from another CMake project

Uni20 can be consumed directly from source using `add_subdirectory` or CMake
FetchContent. It currently does not provide an installed CMake package or
exported targets for `find_package(uni20)`.

## Local checkout

```cmake
cmake_minimum_required(VERSION 3.24)
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

Uni20 leaves the parent's build type, configuration list, compiler flags, and
BLAS selection hints intact. Uni20-specific dependency configuration stays in
its directory scope. It defaults to storing dependency sources and builds under
its own binary directory when embedded; standalone builds retain the shared
source-cache default. `UNI20_FETCHCONTENT_BASE_DIR`,
`UNI20_FETCHCONTENT_SOURCE`, and `UNI20_FETCHCONTENT_SOURCE_BASE_DIR` remain
available as explicit overrides.

Explicitly enabled developer targets are named `uni20_doc`,
`uni20_clang_format`, and (with GCC coverage enabled) `uni20_coverage` when
embedded. Standalone builds retain `doc`, `clang_format`, and `coverage`.
Their inputs and outputs refer to Uni20's own source and binary directories.
`UNI20_DOCS_WEB=ON` always enables the documentation target.
