foreach(case IN ITEMS missing broken)
  set(package_paths)
  if(case STREQUAL "broken")
    list(APPEND package_paths "-DUNI20_GMP_INCLUDE_DIR=${gmp_include}"
      "-DUNI20_GMP_LIBRARY=${gmp_library}" "-DUNI20_MPFR_INCLUDE_DIR=${mpfr_include}")
  endif()
  execute_process(COMMAND "${CMAKE_COMMAND}" --fresh
    -S "${UNI20_SOURCE_DIR}/tests/cmake/mpfr" -B "${test_binary_dir}/${case}"
    "-DUNI20_SOURCE_DIR=${UNI20_SOURCE_DIR}" "-Dcase=${case}"
    "-DCMAKE_CXX_COMPILER=${compiler}"
    ${package_paths}
    RESULT_VARIABLE result OUTPUT_VARIABLE output ERROR_VARIABLE error)
  if(result EQUAL 0)
    message(FATAL_ERROR "MPFR dependency case ${case} unexpectedly configured")
  endif()
  if(case STREQUAL "missing")
    set(expected "requires installed GMP and MPFR development packages")
  else()
    set(expected "Cannot compile and link MPFR")
  endif()
  if(NOT "${output}${error}" MATCHES "${expected}")
    message(FATAL_ERROR "Unexpected failure for ${case}:\n${output}\n${error}")
  endif()
endforeach()
