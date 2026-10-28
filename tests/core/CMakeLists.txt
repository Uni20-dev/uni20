# tests/core/CMakeLists.txt

add_test_module(complex_log_gamma SOURCES test_complex_log_gamma.cpp LIBS uni20_core uni20_async TBB::tbb)
add_test_module(finite_quadrature SOURCES test_finite_quadrature.cpp LIBS uni20_core)

if(UNI20_HAS_MPREAL)
  # Compile the real/complex dispatch headers with UBSan even in ordinary CI
  # builds. No executable or sanitizer runtime is needed for this regression.
  if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    add_library(uni20_mpfr_ubsan_compile OBJECT test_mpfr_ubsan_compile.cpp)
    target_link_libraries(uni20_mpfr_ubsan_compile PRIVATE uni20_core)
    target_compile_options(uni20_mpfr_ubsan_compile PRIVATE -fsanitize=undefined)
  endif()
  add_test_module(mpfr_conversion SOURCES test_mpreal_native_conversion.cpp LIBS uni20_core)
  add_test_module(mpfr SOURCES test_mpreal.cpp test_mpreal_elementary.cpp test_mpreal_utilities.cpp test_mpreal_special.cpp test_mpreal_extensions.cpp test_exact_rational.cpp test_scaled_rational.cpp LIBS uni20_core uni20_async TBB::tbb)
endif()

if(UNI20_HAS_MPREAL)
  add_test_module(mpc SOURCES test_mpcomplex.cpp LIBS uni20_core uni20_async TBB::tbb)
endif()

add_test_module(core
  SOURCES test_types.cpp test_numeric_limits.cpp test_precision.cpp test_scalar_traits.cpp test_scalar_concepts.cpp test_scalar_io.cpp
          test_scalar_precision.cpp test_float80.cpp test_math.cpp test_scalar_math.cpp test_math_utilities.cpp test_native_special_math.cpp test_native_pi_math.cpp test_native_exponential_math.cpp test_native_compound_math.cpp test_native_reciprocal_math.cpp test_native_agm_math.cpp test_factorial.cpp test_native_digamma.cpp test_native_beta.cpp test_native_dilog.cpp test_native_expint.cpp test_native_zeta.cpp test_native_upper_gamma.cpp test_native_airy.cpp test_math_constants.cpp test_buildinfo.cpp
  LIBS uni20_core
)

add_executable(uni20_native_special_availability test_native_special_availability.cpp)
target_link_libraries(uni20_native_special_availability PRIVATE uni20_core)
add_test(NAME NativeSpecialMath.ProviderAvailability COMMAND uni20_native_special_availability)

if(UNI20_HAS_MPLAPACK_FLOAT128)
  add_executable(uni20_mplapack_binary128_tests
    test_mplapack_binary128.cpp
    "${PROJECT_SOURCE_DIR}/tests/uni20_tests_main.cpp"
  )
  target_link_libraries(uni20_mplapack_binary128_tests
    PRIVATE
      uni20_core
      uni20_backend_blas
      mplapack::mplapack_binary128
      GTest::gtest
  )
  target_compile_definitions(uni20_mplapack_binary128_tests PRIVATE UNIT_TEST)
  gtest_discover_tests(uni20_mplapack_binary128_tests)
endif()

add_test_module(elliptic_theta SOURCES test_elliptic_theta.cpp LIBS uni20_core)
