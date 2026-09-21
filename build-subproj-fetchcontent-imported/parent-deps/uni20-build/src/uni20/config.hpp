#pragma once

#include <cstdint>

#define UNI20_ENABLE_STACKTRACE 0

#define STRINGIFY(x) #x

#if UNI20_ENABLE_STACKTRACE
  #include <version>
  #if defined(__cpp_lib_stacktrace) && (__cpp_lib_stacktrace >= 202011L)
    #define UNI20_HAS_STACKTRACE 1
  #else
    #define UNI20_HAS_STACKTRACE 0
  #endif
#else
  #define UNI20_HAS_STACKTRACE 0
#endif

// Automatically generated trace module enable flags:
#define ENABLE_TRACE_BLAS 0
#define ENABLE_TRACE_LAPACK 0
#define ENABLE_TRACE_CUBLAS 0
#define ENABLE_TRACE_CUSOLVER 0
#define ENABLE_TRACE_ASYNC 0
#define ENABLE_TRACE_TESTMODULE 1


// Debugging options
#define UNI20_DEBUG_DAG 0
#define UNI20_DEBUG_ASYNC_TASKS 0
#define UNI20_ASYNC_DEBUG 0

#define UNI20_FALLBACK_TERMINAL_WIDTH 132

// Backend configurations
#define UNI20_BACKEND_BLAS 1
#define UNI20_BACKEND_MKL 0
#define UNI20_BACKEND_MKL_SEQUENTIAL 0
#define UNI20_BACKEND_MKL_THREADED 0
#define UNI20_BACKEND_CUDA 0
#define UNI20_BACKEND_CUBLAS 0
#define UNI20_BACKEND_CUSOLVER 0
#define UNI20_BACKEND_OPENBLAS 0

#define UNI20_ILP64 0

// Optional scalar precision configuration.
#define UNI20_HAS_FLOAT128 0
#define UNI20_FLOAT128_PROVIDER_MPLAPACK 0

// Use the Kokkos reference mdspan implementation in the Uni20 extension namespace.
#ifndef MDSPAN_IMPL_STANDARD_NAMESPACE
#define MDSPAN_IMPL_STANDARD_NAMESPACE stdex
#endif

#define UNI20_BLAS_VENDOR "Generic"

#define UNI20_BLAS_VENDOR_GENERIC 1

namespace uni20 {

#if 4 == 8
using blas_int = std::int64_t;
#else
using blas_int = std::int32_t;
#endif

} // namespace uni20
