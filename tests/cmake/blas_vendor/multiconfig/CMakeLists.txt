cmake_minimum_required(VERSION 3.24)
project(blas_vendor_multiconfig LANGUAGES NONE)
set(CMAKE_CONFIGURATION_TYPES "Debug;Release" CACHE STRING "Configurations under test" FORCE)
add_library(BLAS::BLAS UNKNOWN IMPORTED)
set_target_properties(BLAS::BLAS PROPERTIES IMPORTED_CONFIGURATIONS "DEBUG;RELEASE"
  IMPORTED_LOCATION_DEBUG "${CMAKE_CURRENT_BINARY_DIR}/libopenblas-debug.so"
  IMPORTED_LOCATION_RELEASE "${CMAKE_CURRENT_BINARY_DIR}/libopenblas.so")
include("${UNI20_SOURCE_DIR}/cmake/DetectBlasVendor.cmake")
detect_blas_vendor()
if(NOT UNI20_DETECTED_BLAS_VENDOR STREQUAL "OpenBLAS")
  message(FATAL_ERROR "Same provider in both configurations must identify OpenBLAS")
endif()
set_property(TARGET BLAS::BLAS PROPERTY IMPORTED_LOCATION_RELEASE
  "${CMAKE_CURRENT_BINARY_DIR}/libmkl_rt.so")
detect_blas_vendor()
if(NOT UNI20_DETECTED_BLAS_VENDOR STREQUAL "Generic")
  message(FATAL_ERROR "Different configured providers cannot enable one shared vendor extension")
endif()
