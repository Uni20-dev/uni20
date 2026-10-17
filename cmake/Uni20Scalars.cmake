# Scalar families are user choices; provider capabilities are derived results.
include_guard(GLOBAL)

foreach(_obsolete IN ITEMS UNI20_ENABLE_MPFR UNI20_ENABLE_MPC
    UNI20_ENABLE_MPLAPACK UNI20_ENABLE_MPLAPACK_BINARY80 UNI20_ENABLE_MPLAPACK_MPFR)
  if(DEFINED ${_obsolete})
    message(FATAL_ERROR
      "${_obsolete} has been removed. Select UNI20_ENABLE_FLOAT80, "
      "UNI20_ENABLE_FLOAT128 and UNI20_ENABLE_MPREAL (AUTO/ON/OFF), and "
      "UNI20_BACKEND_MPLAPACK (ON/OFF). Remove old entries from the cache; "
      "see docs/linalg/scalar_configuration.md.")
  endif()
endforeach()

function(uni20_scalar_option name default_value description)
  if(DEFINED ${name})
    string(TOUPPER "${${name}}" _mode)
  else()
    set(_mode "${default_value}")
  endif()
  if(NOT _mode MATCHES "^(AUTO|ON|OFF)$")
    message(FATAL_ERROR "${name}='${${name}}': expected AUTO, ON or OFF")
  endif()
  set(${name} "${_mode}" CACHE STRING
    "${description}: ON (required), OFF (disabled), AUTO (detect without fetching)" FORCE)
  set_property(CACHE ${name} PROPERTY STRINGS AUTO ON OFF)
  # Honor a parent project's normal variable without leaving it unnormalized.
  set(${name} "${_mode}" PARENT_SCOPE)
endfunction()

uni20_scalar_option(UNI20_ENABLE_FLOAT80 AUTO "Native float80 and complex160 scalars")
uni20_scalar_option(UNI20_ENABLE_FLOAT128 OFF "Binary128 float128 and complex256 scalars")
uni20_scalar_option(UNI20_ENABLE_MPREAL OFF "Arbitrary-precision mpreal and mpcomplex scalars")
option(UNI20_BACKEND_MPLAPACK "Enable MPLAPACK kernels for configured scalar families" OFF)

foreach(_capability IN ITEMS FLOAT80 FLOAT128 MPREAL MPLAPACK_FLOAT80
    MPLAPACK_FLOAT128 MPLAPACK_MPREAL MPLAPACK_FLOAT80_COMPLEX)
  set(UNI20_HAS_${_capability} OFF)
endforeach()
set(UNI20_FLOAT128_PROVIDER_MPLAPACK OFF)
foreach(_probe IN ITEMS MPFR_LINKS MPFR_HAS_TLS MPC_LINKS MPFR_FLOAT128_LINKS
    MPLAPACK_BINARY128_HEADERS_COMPATIBLE MPLAPACK_BINARY80_HEADERS_COMPATIBLE
    MPLAPACK_BINARY80_COMPLEX_USES_LONG_DOUBLE DETECTED_FLOAT80 DETECTED_FLOAT128 DETECTED_MPREAL)
  unset(UNI20_${_probe} CACHE)
endforeach()

# Scalar-only binary128 still uses MPLAPACK's scalar headers and math today.
# Test the same public arithmetic surface before advertising that family.
function(_uni20_check_float128 output)
  include(CheckCXXSourceCompiles)
  block(SCOPE_FOR VARIABLES)
    set(CMAKE_TRY_COMPILE_TARGET_TYPE STATIC_LIBRARY)
    if(UNI20_MPLAPACK_SOURCE STREQUAL "fetched")
      get_target_property(_uni20_mplapack_binary128_target mplapack::mplapack_binary128 ALIASED_TARGET)
      get_target_property(_uni20_mplapack_source_dir ${_uni20_mplapack_binary128_target} SOURCE_DIR)
      get_target_property(_uni20_mplapack_binary_dir ${_uni20_mplapack_binary128_target} BINARY_DIR)
      set(CMAKE_REQUIRED_DEFINITIONS -DMPLAPACK_BUILD_WITH_BINARY128)
      set(CMAKE_REQUIRED_INCLUDES
          "${_uni20_mplapack_binary_dir}/include"
          "${_uni20_mplapack_source_dir}/include")
    else()
      set(CMAKE_REQUIRED_LIBRARIES mplapack::mplapack_binary128)
    endif()
    unset(UNI20_MPLAPACK_BINARY128_HEADERS_COMPATIBLE CACHE)
    check_cxx_source_compiles(
      [[
        #include <complex>
        #include <cmath>
        #include <cstdlib>
        #include <mpblas_binary128.h>
        #include <mplapack_binary128.h>

        #ifndef MPLAPACK_BINARY128_MODE
        #error "MPLAPACK_BINARY128_MODE is not defined"
        #endif

        #if MPLAPACK_BINARY128_MODE == MPLAPACK_BINARY128_MODE_LDBL
        #error "MPLAPACK binary128 aliases long double"
        #endif

        int main()
        {
          mplapack_binary128_t x = static_cast<mplapack_binary128_t>(1);
          auto y = abs(x);
          std::complex<mplapack_binary128_t> z{x, x};
          auto a = abs(z);
          auto b = sqrt(z);
          (void)y;
          (void)a;
          (void)b;
          return 0;
        }
      ]]
      UNI20_MPLAPACK_BINARY128_HEADERS_COMPATIBLE)
  endblock()
  set(_available OFF)
  if(UNI20_MPLAPACK_BINARY128_HEADERS_COMPATIBLE)
    set(_available ON)
  endif()
  set(${output} ${_available} PARENT_SCOPE)
endfunction()

macro(uni20_configure_scalar_providers)
  include(CheckCXXSourceCompiles)
  if(NOT UNI20_ENABLE_FLOAT80 STREQUAL "OFF")
    unset(UNI20_NATIVE_FLOAT80 CACHE)
    check_cxx_source_compiles([=[
      #include <limits>
      using L = std::numeric_limits<long double>;
      static_assert(L::is_iec559 && L::radix == 2 && L::digits == 64);
      static_assert(L::min_exponent == -16381 && L::max_exponent == 16384);
      static_assert(L::denorm_min() > 0 && L::denorm_min() < L::min());
      int main() {}
    ]=] UNI20_NATIVE_FLOAT80)
    if(UNI20_ENABLE_FLOAT80 STREQUAL "ON" AND NOT UNI20_NATIVE_FLOAT80)
      message(FATAL_ERROR "UNI20_ENABLE_FLOAT80=ON requires native x87 extended-format long double")
    endif()
    if(UNI20_NATIVE_FLOAT80)
      set(UNI20_HAS_FLOAT80 ON)
    endif()
  endif()

  if(NOT UNI20_ENABLE_MPREAL STREQUAL "OFF")
    include(Uni20MPFR)
    include(Uni20MPC)
    set(_uni20_mpreal_required OFF)
    if(UNI20_ENABLE_MPREAL STREQUAL "ON")
      set(_uni20_mpreal_required ON)
    endif()
    uni20_find_mpfr(${_uni20_mpreal_required})
    if(UNI20_MPFR_FOUND)
      uni20_find_mpc(${_uni20_mpreal_required})
      if(UNI20_MPC_FOUND)
        set(UNI20_HAS_MPREAL ON)
      endif()
    endif()
  endif()

  set(_uni20_scalar_system_mode AUTO)
  if(DEFINED UNI20_USE_SYSTEM_MPLAPACK)
    _uni20_normalize_dependency_mode("${UNI20_USE_SYSTEM_MPLAPACK}" _uni20_scalar_system_mode)
  endif()
  set(_uni20_want_float128 OFF)
  if(UNI20_ENABLE_FLOAT128 STREQUAL "ON")
    set(_uni20_want_float128 ON)
  elseif(UNI20_ENABLE_FLOAT128 STREQUAL "AUTO" AND NOT _uni20_scalar_system_mode STREQUAL "OFF")
    # AUTO never downloads a scalar provider. A parent-supplied target also counts.
    if(NOT TARGET mplapack::mplapack_binary128)
      find_package(mplapack 3.0.0 QUIET CONFIG NO_CMAKE_PACKAGE_REGISTRY COMPONENTS binary128)
    endif()
    if(TARGET mplapack::mplapack_binary128)
      set(UNI20_MPLAPACK_SOURCE system)
      _uni20_check_float128(_uni20_want_float128)
    endif()
  endif()

  if(_uni20_want_float128 AND UNI20_HAS_MPREAL)
    set(_uni20_conversions_required OFF)
    if(UNI20_ENABLE_FLOAT128 STREQUAL "ON" AND UNI20_ENABLE_MPREAL STREQUAL "ON")
      set(_uni20_conversions_required ON)
    endif()
    uni20_enable_mpfr_float128(${_uni20_conversions_required})
    if(NOT UNI20_MPFR_FLOAT128_FOUND)
      if(UNI20_ENABLE_FLOAT128 STREQUAL "AUTO")
        set(_uni20_want_float128 OFF)
      else()
        set(UNI20_HAS_MPREAL OFF)
      endif()
    endif()
  endif()

  set(_uni20_mplapack_components "")
  if(_uni20_want_float128)
    list(APPEND _uni20_mplapack_components binary128)
  endif()
  if(UNI20_BACKEND_MPLAPACK)
    if(UNI20_HAS_FLOAT80)
      set(UNI20_HAS_MPLAPACK_FLOAT80 ON)
      list(APPEND _uni20_mplapack_components binary80)
    endif()
    if(UNI20_HAS_MPREAL)
      set(UNI20_HAS_MPLAPACK_MPREAL ON)
      list(APPEND _uni20_mplapack_components mpfr)
    endif()
  endif()

  # Clear this configure's resolution, not the user's package search hints.
  foreach(_result IN ITEMS DIR SOURCE TARGET VERSION)
    unset(UNI20_MPLAPACK_${_result} CACHE)
    unset(UNI20_MPLAPACK_${_result})
  endforeach()
  unset(UNI20_DETECTED_MPLAPACK CACHE)
  if(_uni20_mplapack_components)
    uni20_dependency_option(UNI20_USE_SYSTEM_MPLAPACK "MPLAPACK library" AUTO)
    list(GET _uni20_mplapack_components 0 _uni20_mplapack_primary)
    uni20_add_dependency(
      NAME mplapack
      VERSION 3.0.0
      TARGET mplapack::mplapack_${_uni20_mplapack_primary}
      REPO https://github.com/nakatamaho/mplapack.git
      TAG v3.0.0
      COMPONENTS ${_uni20_mplapack_components}
      SETTINGS
        "MPLAPACK_CXX_STANDARD=23"
        "MPLAPACK_CXX_EXTENSIONS=ON"
        "MPLAPACK_ENABLE_BINARY128=${_uni20_want_float128}"
        "MPLAPACK_ENABLE_MPFR=${UNI20_HAS_MPLAPACK_MPREAL}"
        "MPLAPACK_ENABLE_BINARY80=${UNI20_HAS_MPLAPACK_FLOAT80}"
        "MPLAPACK_ENABLE_OPT=OFF"
        "MPLAPACK_ENABLE_CUDA=OFF"
        "MPLAPACK_ENABLE_OPENCL=OFF"
      DEFAULTS
        "MPLAPACK_ENABLE_DOUBLE=OFF"
        "MPLAPACK_ENABLE_GMP=OFF"
        "MPLAPACK_ENABLE_QD=OFF"
        "MPLAPACK_ENABLE_DD=OFF"
        "MPLAPACK_BUILD_EXAMPLES=OFF"
        "MPLAPACK_BUILD_TESTS=OFF"
        "MPLAPACK_BUILD_BENCHMARKS=OFF"
    )
    foreach(_uni20_component IN LISTS _uni20_mplapack_components)
      if(NOT TARGET mplapack::mplapack_${_uni20_component})
        message(FATAL_ERROR "MPLAPACK must provide target mplapack::mplapack_${_uni20_component}")
      endif()
      if(UNI20_MPLAPACK_SOURCE STREQUAL "fetched")
        relax_warnings(mplapack::mplapack_${_uni20_component})
        if(_uni20_component STREQUAL "binary80" AND CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
          # Upstream uses the GNU f64x literal suffix in its arithmetic limits.
          # -Wno-error alone does not override our inherited -pedantic-errors.
          get_target_property(_uni20_binary80_target mplapack::mplapack_binary80 ALIASED_TARGET)
          target_compile_options(${_uni20_binary80_target} PRIVATE -Wno-error=pedantic)
        endif()
      endif()
    endforeach()
  endif()

  if(_uni20_want_float128)
    _uni20_check_float128(UNI20_HAS_FLOAT128)
    if(NOT UNI20_HAS_FLOAT128)
      message(FATAL_ERROR
        "UNI20_ENABLE_FLOAT128 requires an MPLAPACK binary128 scalar provider "
        "compatible with this C++23 toolchain. See docs/linalg/mplapack_binary128.md.")
    endif()
    set(UNI20_FLOAT128_PROVIDER_MPLAPACK ON)
    if(UNI20_BACKEND_MPLAPACK)
      set(UNI20_HAS_MPLAPACK_FLOAT128 ON)
    endif()
  endif()

  if(UNI20_HAS_MPLAPACK_FLOAT80)
    include(CheckCXXSourceCompiles)
    block(SCOPE_FOR VARIABLES)
      set(CMAKE_TRY_COMPILE_TARGET_TYPE STATIC_LIBRARY)
      if(UNI20_MPLAPACK_SOURCE STREQUAL "fetched")
        get_target_property(_target mplapack::mplapack_binary80 ALIASED_TARGET)
        get_target_property(_source ${_target} SOURCE_DIR)
        get_target_property(_binary ${_target} BINARY_DIR)
        set(CMAKE_REQUIRED_DEFINITIONS -DMPLAPACK_BUILD_WITH_BINARY80)
        set(CMAKE_REQUIRED_INCLUDES "${_binary}/include" "${_source}/include")
      else()
        set(CMAKE_REQUIRED_LIBRARIES mplapack::mplapack_binary80)
      endif()
      unset(UNI20_MPLAPACK_BINARY80_HEADERS_COMPATIBLE CACHE)
      check_cxx_source_compiles([=[
        #include <complex>
        #include <limits>
        #include <mpblas_binary80.h>
        #include <mplapack_binary80.h>
        #if MPLAPACK_BINARY80_MODE == MPLAPACK_BINARY80_MODE_FLOAT64X
        static_assert(__FLT64X_MANT_DIG__ == 64 && __FLT64X_MAX_EXP__ == 16384);
        #elif MPLAPACK_BINARY80_MODE == MPLAPACK_BINARY80_MODE_LDBL80
        static_assert(std::numeric_limits<long double>::digits == 64 &&
                      std::numeric_limits<long double>::max_exponent == 16384);
        #else
        #error "Unsupported MPLAPACK binary80 scalar format"
        #endif
        int main() {
          mplapack_binary80_t a = static_cast<mplapack_binary80_t>(1);
          std::complex<mplapack_binary80_t> z{a, a};
          long double round_trip = static_cast<long double>(a);
          a = static_cast<mplapack_binary80_t>(round_trip);
          mplapackint pivot = 0, info = 0;
          Rgetrf(1, 1, &a, 1, &pivot, info);
          Cgetrf(1, 1, &z, 1, &pivot, info);
        }
      ]=] UNI20_MPLAPACK_BINARY80_HEADERS_COMPATIBLE)
      # Generic std::complex<_Float64x> division can square its denominator,
      # overflowing or underflowing on otherwise representable LU problems.
      # Only enable complex LU/decompositions for the native long-double specialization.
      # This is a compile-time capability check, including for cross builds.
      unset(UNI20_MPLAPACK_BINARY80_COMPLEX_USES_LONG_DOUBLE CACHE)
      check_cxx_source_compiles([=[
        #include <type_traits>
        #include <mpblas_binary80.h>
        static_assert(std::is_same_v<mplapack_binary80_t, long double>);
        int main() {}
      ]=] UNI20_MPLAPACK_BINARY80_COMPLEX_USES_LONG_DOUBLE)
    endblock()
    if(NOT UNI20_MPLAPACK_BINARY80_HEADERS_COMPATIBLE)
      message(FATAL_ERROR "MPLAPACK binary80 headers must support native long double value conversion and real/complex LU")
    endif()
    set(UNI20_HAS_MPLAPACK_FLOAT80_COMPLEX ${UNI20_MPLAPACK_BINARY80_COMPLEX_USES_LONG_DOUBLE})
    message(STATUS "MPLAPACK binary80 backend enabled (${UNI20_MPLAPACK_SOURCE})")
    if(NOT UNI20_HAS_MPLAPACK_FLOAT80_COMPLEX)
      message(STATUS "MPLAPACK binary80 complex LU/decompositions disabled: provider scalar is not native long double; default complex LU uses CPU")
    endif()
  endif()

  foreach(_family IN ITEMS FLOAT80 FLOAT128 MPREAL)
    set(_provider none)
    if(UNI20_HAS_${_family})
      if(_family STREQUAL "FLOAT80")
        set(_provider "native long double")
      elseif(_family STREQUAL "FLOAT128")
        set(_provider "MPLAPACK scalar support")
      else()
        set(_provider "MPFR + MPC + GMP")
      endif()
    endif()
    message(STATUS "${_family}: requested=${UNI20_ENABLE_${_family}}, available=${UNI20_HAS_${_family}}, arithmetic=${_provider}, MPLAPACK kernels=${UNI20_HAS_MPLAPACK_${_family}}")
    set(UNI20_DETECTED_${_family}
      "${_provider}; MPLAPACK kernels=${UNI20_HAS_MPLAPACK_${_family}}"
      CACHE STRING "Resolved scalar arithmetic and linear-algebra provider" FORCE)
  endforeach()
  if(UNI20_BACKEND_MPLAPACK AND NOT _uni20_mplapack_components)
    message(FATAL_ERROR "UNI20_BACKEND_MPLAPACK=ON requires at least one available optional scalar family")
  endif()
endmacro()
