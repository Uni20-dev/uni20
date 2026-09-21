# Directory-local imported targets let each case exercise the real detector.
if(vendor_case MATCHES "^implib")
  add_library(BLAS::BLAS SHARED IMPORTED)
else()
  add_library(BLAS::BLAS UNKNOWN IMPORTED)
endif()
set(CMAKE_BUILD_TYPE RelWithDebInfo)
set(BLA_VENDOR All)
set(BLAS_LIBRARIES "/unrelated/libmkl_rt.so")
set(expected_vendor OpenBLAS)
set(openblas "${CMAKE_CURRENT_BINARY_DIR}/libopenblas.so")
set(other "${CMAKE_CURRENT_BINARY_DIR}/libmkl_rt.so")

if(vendor_case STREQUAL "plain")
  set_target_properties(BLAS::BLAS PROPERTIES IMPORTED_LOCATION "${openblas}")
elseif(vendor_case STREQUAL "exact")
  set_target_properties(BLAS::BLAS PROPERTIES
    IMPORTED_LOCATION "${other}" IMPORTED_LOCATION_RELWITHDEBINFO "${openblas}")
elseif(vendor_case STREQUAL "mapped")
  set_target_properties(BLAS::BLAS PROPERTIES
    IMPORTED_CONFIGURATIONS "DEBUG;RELEASE"
    IMPORTED_LOCATION_DEBUG "${other}" IMPORTED_LOCATION_RELEASE "${openblas}"
    IMPORTED_LOCATION "${other}"
    MAP_IMPORTED_CONFIG_RELWITHDEBINFO "Missing;Release")
elseif(vendor_case MATCHES "^mapped_.*empty$")
  set_target_properties(BLAS::BLAS PROPERTIES
    IMPORTED_CONFIGURATIONS RELEASE IMPORTED_LOCATION_RELEASE "${other}"
    IMPORTED_LOCATION "${openblas}" MAP_IMPORTED_CONFIG_RELWITHDEBINFO "")
  if(vendor_case STREQUAL "mapped_list_empty")
    set_property(TARGET BLAS::BLAS PROPERTY MAP_IMPORTED_CONFIG_RELWITHDEBINFO "Missing;;Release")
  endif()
elseif(vendor_case STREQUAL "default_location")
  set_target_properties(BLAS::BLAS PROPERTIES
    IMPORTED_CONFIGURATIONS RELEASE IMPORTED_LOCATION_RELEASE "${other}"
    IMPORTED_LOCATION "${openblas}")
elseif(vendor_case STREQUAL "fallback")
  set_target_properties(BLAS::BLAS PROPERTIES
    IMPORTED_CONFIGURATIONS "RELEASE;DEBUG"
    IMPORTED_LOCATION_RELEASE "${openblas}" IMPORTED_LOCATION_DEBUG "${other}")
elseif(vendor_case STREQUAL "no_config")
  set(CMAKE_BUILD_TYPE "")
  set_target_properties(BLAS::BLAS PROPERTIES IMPORTED_LOCATION_NOCONFIG "${openblas}")
elseif(vendor_case STREQUAL "implib")
  set_target_properties(BLAS::BLAS PROPERTIES
    IMPORTED_LOCATION "${CMAKE_CURRENT_BINARY_DIR}/provider.dll"
    IMPORTED_IMPLIB "${CMAKE_CURRENT_BINARY_DIR}/openblas.lib")
elseif(vendor_case MATCHES "^implib_")
  set_target_properties(BLAS::BLAS PROPERTIES
    IMPORTED_CONFIGURATIONS "DEBUG;RELEASE"
    IMPORTED_IMPLIB_DEBUG "${other}"
    IMPORTED_IMPLIB_RELEASE "${CMAKE_CURRENT_BINARY_DIR}/openblas.lib"
    MAP_IMPORTED_CONFIG_RELWITHDEBINFO Release)
  if(vendor_case STREQUAL "implib_mapped")
    set_property(TARGET BLAS::BLAS PROPERTY IMPORTED_LOCATION_RELEASE
      "${CMAKE_CURRENT_BINARY_DIR}/provider.dll")
  endif()
elseif(vendor_case STREQUAL "opaque")
  set_target_properties(BLAS::BLAS PROPERTIES IMPORTED_LOCATION "${CMAKE_CURRENT_BINARY_DIR}/provider.so")
  set(expected_vendor Generic)
endif()

include("${UNI20_SOURCE_DIR}/cmake/DetectBlasVendor.cmake")
detect_blas_vendor()
if(NOT UNI20_DETECTED_BLAS_VENDOR STREQUAL expected_vendor)
  message(FATAL_ERROR "${vendor_case}: expected ${expected_vendor}, got ${UNI20_DETECTED_BLAS_VENDOR}")
endif()

# CMake's own resolved location is an independent oracle for configuration
# selection. Import-library-only targets have no runtime file to query.
if(NOT vendor_case STREQUAL "implib_only")
  if(CMAKE_BUILD_TYPE)
    get_target_property(selected BLAS::BLAS "LOCATION_${CMAKE_BUILD_TYPE}")
  else()
    get_target_property(selected BLAS::BLAS LOCATION)
  endif()
  if(NOT selected IN_LIST UNI20_DETECTED_BLAS_LIBRARIES)
    message(FATAL_ERROR "${vendor_case}: detection omitted CMake-selected file ${selected}")
  endif()
endif()
