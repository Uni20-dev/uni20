# LSan subprocess controls intentionally fail on real leaks. Other builds do
# not register these checks; Linux is the validated LSan runtime here.
if(CMAKE_SYSTEM_NAME STREQUAL "Linux" AND UNI20_HAS_MPREAL
   AND ("address" IN_LIST UNI20_SANITIZE_LIST OR "leak" IN_LIST UNI20_SANITIZE_LIST))
  add_executable(uni20_mpfr_cache_probe mpfr_cache_probe.cpp)
  target_link_libraries(uni20_mpfr_cache_probe PRIVATE uni20_core)
  # Keep the owning-result allocation stack visible, including the templated
  # apply<&mpfr_atan> frame that an unanchored suppression would also match.
  target_compile_options(uni20_mpfr_cache_probe PRIVATE -O1 -g1)
  add_test(NAME Sanitizers.MpfrCachePolicy
    COMMAND "${CMAKE_COMMAND}"
      "-Dprobe=$<TARGET_FILE:uni20_mpfr_cache_probe>"
      "-Dsuppressions=${CMAKE_CURRENT_SOURCE_DIR}/lsan.supp"
      "-Dtest_binary_dir=${CMAKE_CURRENT_BINARY_DIR}"
      -P "${CMAKE_CURRENT_SOURCE_DIR}/check_mpfr_cache.cmake")
  set_tests_properties(Sanitizers.MpfrCachePolicy PROPERTIES
    LABELS sanitizers TIMEOUT 120)
endif()
