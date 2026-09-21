#pragma once

#include <array>
#include <span>
#include <string_view>

namespace uni20::build_info {

struct Entry {
  std::string_view key;
  std::string_view value;
  std::string_view help;
};

struct Info {
  std::string_view generator;
  std::string_view build_type;
  std::string_view system_name;
  std::string_view system_version;
  std::string_view system_processor;
  std::string_view cxx_compiler_id;
  std::string_view cxx_compiler_version;
  std::string_view cxx_compiler_path;
  std::span<Entry const> build_options;
  std::span<Entry const> detected_environment;
};

inline constexpr std::string_view kGenerator{"Unix Makefiles"};
inline constexpr std::string_view kBuildType{"Multi-config (Debug, Release, ConsumerSpecial)"};
inline constexpr std::string_view kSystemName{"Linux"};
inline constexpr std::string_view kSystemVersion{"6.17.0-1022-azure"};
inline constexpr std::string_view kSystemProcessor{"x86_64"};
inline constexpr std::string_view kCompilerId{"GNU"};
inline constexpr std::string_view kCompilerVersion{"13.3.0"};
inline constexpr std::string_view kCompilerPath{"/usr/bin/c++"};

inline constexpr std::array<Entry, 59> kBuildOptions{{
    {"UNI20_ASYNC_DEBUG", "OFF", "Enable Async debug counters and tracing"},
    {"UNI20_BACKEND_BLAS", "OFF", "Enable the BLAS backend"},
    {"UNI20_BACKEND_CUBLAS", "OFF", "Enable the cuBLAS backend"},
    {"UNI20_BACKEND_CUDA", "OFF", "Enable the CUDA backend"},
    {"UNI20_BACKEND_CUSOLVER", "OFF", "Enable the cuSOLVER backend"},
    {"UNI20_BACKEND_MKL", "OFF", "Enable the compatibility MKL backend extension API selector"},
    {"UNI20_BACKEND_MKL_SEQUENTIAL", "OFF", "Enable the sequential MKL backend extension API"},
    {"UNI20_BACKEND_MKL_THREADED", "OFF", "Enable the threaded MKL backend extension API"},
    {"UNI20_BACKEND_OPENBLAS", "OFF", "Enable the OpenBLAS backend extension API"},
    {"UNI20_BLAS_VENDOR", "All", "BLAS vendor hint passed to FindBLAS as BLA_VENDOR (for example: All, OpenBLAS, Generic, Intel10_64lp_seq, Intel10_64_dyn)"},
    {"UNI20_BUILD_ASM", "OFF", "Build stand-alone assembly dumps for snippets in asm/"},
    {"UNI20_BUILD_BENCH", "OFF", "Build benchmarks"},
    {"UNI20_BUILD_COMBINED_TESTS", "OFF", "Build combined test executable (in addition to per-module tests)"},
    {"UNI20_BUILD_DOCS", "OFF", "Add the Doxygen documentation target"},
    {"UNI20_BUILD_EXAMPLES", "OFF", "Build the example programs"},
    {"UNI20_BUILD_EXTERNAL_TESTS", "OFF", "Build tests for external dependencies"},
    {"UNI20_BUILD_FORMAT_TARGET", "OFF", "Add the clang-format target"},
    {"UNI20_BUILD_PYTHON", "OFF", "Build the Python bindings"},
    {"UNI20_BUILD_TESTS", "OFF", "Build unit tests"},
    {"UNI20_DEBUG_ASYNC_TASKS", "OFF", "Enable AsyncTask debug instrumentation"},
    {"UNI20_DEBUG_DAG", "OFF", "Enable DAG debug info (for Async buffer tracking and visualization)"},
    {"UNI20_DOCS_WEB", "OFF", "Enable web-oriented Doxygen configuration for deployment"},
    {"UNI20_ENABLE_COVERAGE", "OFF", "Enable code coverage instrumentation"},
    {"UNI20_ENABLE_CUDA", "OFF", "Enable CUDA backend support"},
    {"UNI20_ENABLE_LTO", "ON", "Enable LTO/IPO for optimized builds"},
    {"UNI20_ENABLE_MPI", "OFF", "Enable MPI support"},
    {"UNI20_ENABLE_MPLAPACK", "OFF", "Enable the optional MPLAPACK binary128 backend"},
    {"UNI20_ENABLE_STACKTRACE", "OFF", "Enable C++23 <stacktrace> support when available"},
    {"UNI20_ENABLE_TRACE_ASYNC", "OFF", "Enable tracing for ASYNC"},
    {"UNI20_ENABLE_TRACE_BLAS", "OFF", "Enable tracing for BLAS"},
    {"UNI20_ENABLE_TRACE_CUBLAS", "OFF", "Enable tracing for CUBLAS"},
    {"UNI20_ENABLE_TRACE_CUSOLVER", "OFF", "Enable tracing for CUSOLVER"},
    {"UNI20_ENABLE_TRACE_LAPACK", "OFF", "Enable tracing for LAPACK"},
    {"UNI20_ENABLE_TRACE_TESTMODULE", "ON", "Enable tracing for TESTMODULE"},
    {"UNI20_ENABLE_WARNINGS", "ON", "Enable compiler warnings"},
    {"UNI20_EXTERNAL_NO_WARN", "OFF", "Disable all warnings for external libraries"},
    {"UNI20_EXTERNAL_NO_WERROR", "ON", "Disable -Werror for external libraries"},
    {"UNI20_FALLBACK_TERMINAL_WIDTH", "132", "Terminal width used when COLUMNS and operating-system terminal detection are unavailable"},
    {"UNI20_FETCHCONTENT_BASE_DIR", "/home/runner/work/uni20/uni20/build-subproj-imported-no-blas/uni20/.cmake/third_party", "Resolved FetchContent base directory for build/stamp directories"},
    {"UNI20_FETCHCONTENT_SOURCE", "OFF", "Store FetchContent source trees in a shared source cache directory. OFF keeps sources under UNI20_FETCHCONTENT_BASE_DIR"},
    {"UNI20_FETCHCONTENT_SOURCE_BASE_DIR", "/home/runner/work/uni20/uni20/build-subproj-imported-no-blas/uni20/.cmake/third_party", "Resolved FetchContent source directory. With UNI20_FETCHCONTENT_SOURCE=OFF this matches UNI20_FETCHCONTENT_BASE_DIR"},
    {"UNI20_FMT_SOURCE", "fetched", "Source type for fmt (system or fetched)"},
    {"UNI20_FMT_TARGET", "fmt::fmt", "CMake imported target name for fmt"},
    {"UNI20_FMT_VERSION", "11.1.0", "Requested version for fmt"},
    {"UNI20_ILP64", "OFF", "Enable ILP64 integer type for BLAS (64-bit integers)"},
    {"UNI20_INTEGRATION_MODE", "subdirectory", "subdirectory or fetchcontent"},
    {"UNI20_SANITIZE", "OFF", "Enable sanitizers (comma-separated list: address,undefined,thread,memory,leak)"},
    {"UNI20_SOURCE_DIR", "/home/runner/work/uni20/uni20", "No help, variable specified on the command line."},
    {"UNI20_TBB_SOURCE", "fetched", "Source type for TBB (system or fetched)"},
    {"UNI20_TBB_TARGET", "TBB::tbb", "CMake imported target name for TBB"},
    {"UNI20_TBB_VERSION", "2022.3", "Requested version for TBB"},
    {"UNI20_TEST_DEVELOPER_TARGETS", "OFF", "Opt into embedded documentation and formatting"},
    {"UNI20_TEST_PARENT_BLAS", "imported_no_blas", "No help, variable specified on the command line."},
    {"UNI20_TEST_PYTHON", "OFF", "Build and run embedded Python tests"},
    {"UNI20_USE_SYSTEM_BENCHMARK", "AUTO", "How to resolve Google Benchmark library: AUTO (prefer system, fallback fetch), ON (require system), OFF (always fetch)"},
    {"UNI20_USE_SYSTEM_FMT", "AUTO", "How to resolve fmt library: AUTO (prefer system, fallback fetch), ON (require system), OFF (always fetch)"},
    {"UNI20_USE_SYSTEM_GTEST", "AUTO", "How to resolve Google Test library: AUTO (prefer system, fallback fetch), ON (require system), OFF (always fetch)"},
    {"UNI20_USE_SYSTEM_NANOBIND", "AUTO", "How to resolve nanobind library: AUTO (prefer system, fallback fetch), ON (require system), OFF (always fetch)"},
    {"UNI20_USE_SYSTEM_TBB", "AUTO", "How to resolve oneTBB library: AUTO (prefer system, fallback fetch), ON (require system), OFF (always fetch)"},

}};

inline constexpr std::array<Entry, 3> kDetectedEnvironment{{
    {"UNI20_DETECTED_FMT", "fetched", "Cloned from https://github.com/fmtlib/fmt.git (tag 11.1.4)"},
    {"UNI20_DETECTED_STACKTRACE_PROVIDER", "stdc++exp", "Provider used for std::stacktrace support"},
    {"UNI20_DETECTED_TBB", "fetched", "Cloned from https://github.com/uxlfoundation/oneTBB (tag v2022.3.0)"},

}};

inline constexpr Info current() noexcept
{
  return Info{
      kGenerator,
      kBuildType,
      kSystemName,
      kSystemVersion,
      kSystemProcessor,
      kCompilerId,
      kCompilerVersion,
      kCompilerPath,
      kBuildOptions,
      kDetectedEnvironment,
  };
}

} // namespace uni20::build_info
