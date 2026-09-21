# Configure real parent projects, then compile and run their consumers. These
# tests exercise CMake usage requirements outside Uni20's directory scope.
include(FetchContent)
set(UNI20_SUBPROJECT_TEST_ARGS "")
foreach(dependency IN ITEMS fmt mdspan TBB GTest nanobind)
  unset(dependency_source)
  FetchContent_GetProperties(${dependency} SOURCE_DIR dependency_source)
  if(dependency_source)
    string(TOUPPER "${dependency}" dependency_upper)
    string(APPEND UNI20_SUBPROJECT_TEST_ARGS
      "  [==[-DFETCHCONTENT_SOURCE_DIR_${dependency_upper}=${dependency_source}]==]\n")
  endif()
endforeach()

foreach(variable IN ITEMS CMAKE_CXX_COMPILER CMAKE_C_COMPILER CMAKE_TOOLCHAIN_FILE
    CMAKE_GENERATOR_PLATFORM CMAKE_GENERATOR_TOOLSET CMAKE_GENERATOR_INSTANCE
    CMAKE_MAKE_PROGRAM CMAKE_PREFIX_PATH fmt_DIR TBB_DIR mplapack_DIR
    UNI20_ENABLE_STACKTRACE)
  if(DEFINED ${variable} AND NOT "${${variable}}" MATCHES "-NOTFOUND$")
    string(APPEND UNI20_SUBPROJECT_TEST_ARGS "  [==[-D${variable}=${${variable}}]==]\n")
  endif()
endforeach()

# Do not compile the entire MPLAPACK provider three more times in nested builds.
# Main-suite tests cover the fetched provider; an installed package also gives
# these downstream consumers inexpensive binary128 compile/link/run coverage.
if(UNI20_ENABLE_MPLAPACK AND UNI20_MPLAPACK_SOURCE STREQUAL "system")
  string(APPEND UNI20_SUBPROJECT_TEST_ARGS "  -DUNI20_ENABLE_MPLAPACK=ON\n")
else()
  string(APPEND UNI20_SUBPROJECT_TEST_ARGS "  -DUNI20_ENABLE_MPLAPACK=OFF\n")
endif()

configure_file(run_subproject.cmake.in run_subproject.cmake @ONLY)
add_test(NAME CMakeSubproject.blas_abi_contract
  COMMAND "${CMAKE_COMMAND}" "-DUNI20_SOURCE_DIR=${PROJECT_SOURCE_DIR}"
    "-Dtest_binary_dir=${CMAKE_CURRENT_BINARY_DIR}/blas_abi_contract"
    -P "${CMAKE_CURRENT_SOURCE_DIR}/check_blas_abi.cmake")
foreach(abi_case IN ITEMS matching conflicting imported imported_no_blas)
  add_test(NAME CMakeSubproject.parent_blas_${abi_case}
    COMMAND "${CMAKE_COMMAND}" -Dmode=subdirectory "-Dabi_case=${abi_case}"
      "-Dconfig=$<CONFIG>" -P "${CMAKE_CURRENT_BINARY_DIR}/run_subproject.cmake")
  set_tests_properties(CMakeSubproject.parent_blas_${abi_case} PROPERTIES LABELS cmake TIMEOUT 300)
endforeach()
add_test(NAME CMakeSubproject.dependency_ownership
  COMMAND "${CMAKE_COMMAND}" --fresh
    -S "${CMAKE_CURRENT_SOURCE_DIR}/dependency_ownership"
    -B "${CMAKE_CURRENT_BINARY_DIR}/dependency_ownership"
    "-DUNI20_SOURCE_DIR=${PROJECT_SOURCE_DIR}")
foreach(mode IN ITEMS subdirectory fetchcontent)
  add_test(NAME CMakeSubproject.${mode}
    COMMAND "${CMAKE_COMMAND}" "-Dmode=${mode}" "-Dconfig=$<CONFIG>"
      -P "${CMAKE_CURRENT_BINARY_DIR}/run_subproject.cmake")
  set_tests_properties(CMakeSubproject.${mode} PROPERTIES LABELS cmake TIMEOUT 300)
endforeach()

find_package(Python3 QUIET COMPONENTS Interpreter)
if(UNIX AND Python3_Interpreter_FOUND)
  add_test(NAME CMakeSubproject.developer_targets
    COMMAND "${CMAKE_COMMAND}" -Dmode=subdirectory -Ddeveloper_targets=ON
      "-Dconfig=$<CONFIG>" -P "${CMAKE_CURRENT_BINARY_DIR}/run_subproject.cmake")
  set_tests_properties(CMakeSubproject.developer_targets PROPERTIES LABELS cmake TIMEOUT 300)
endif()
if(UNI20_BUILD_PYTHON)
  add_test(NAME CMakeSubproject.python
    COMMAND "${CMAKE_COMMAND}" -Dmode=subdirectory -Dpython_tests=ON
      "-Dconfig=$<CONFIG>" -P "${CMAKE_CURRENT_BINARY_DIR}/run_subproject.cmake")
  set_tests_properties(CMakeSubproject.python PROPERTIES LABELS cmake TIMEOUT 600)
endif()
