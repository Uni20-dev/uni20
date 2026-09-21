# Configure real parent projects, then compile and run their consumers. These
# tests exercise CMake usage requirements outside Uni20's directory scope.
include(subproject_args.cmake)

# Do not compile the entire MPLAPACK provider three more times in nested builds.
# Main-suite tests cover the fetched provider; an installed package also gives
# these downstream consumers inexpensive binary128 compile/link/run coverage.
if(UNI20_ENABLE_MPLAPACK AND UNI20_MPLAPACK_SOURCE STREQUAL "system")
  string(APPEND UNI20_SUBPROJECT_TEST_ARGS "  -DUNI20_ENABLE_MPLAPACK=ON\n")
else()
  string(APPEND UNI20_SUBPROJECT_TEST_ARGS "  -DUNI20_ENABLE_MPLAPACK=OFF\n")
endif()

configure_file(run_subproject.cmake.in run_subproject.cmake @ONLY)
configure_file(run_dependency_build.cmake.in run_dependency_build.cmake @ONLY)
foreach(external_tests IN ITEMS OFF ON)
  add_test(NAME CMakeSubproject.dependency_build_${external_tests}
    COMMAND "${CMAKE_COMMAND}" "-Dexternal_tests=${external_tests}" "-Dconfig=$<CONFIG>"
      -P "${CMAKE_CURRENT_BINARY_DIR}/run_dependency_build.cmake")
  set_tests_properties(CMakeSubproject.dependency_build_${external_tests}
    PROPERTIES LABELS cmake TIMEOUT 120)
endforeach()
foreach(fixture IN ITEMS blas_vendor package_hints)
  add_test(NAME CMakeSubproject.${fixture}
    COMMAND "${CMAKE_COMMAND}" --fresh
      -S "${CMAKE_CURRENT_SOURCE_DIR}/${fixture}"
      -B "${CMAKE_CURRENT_BINARY_DIR}/${fixture}"
      "-DUNI20_SOURCE_DIR=${PROJECT_SOURCE_DIR}")
endforeach()
if(POLICY CMP0200)
  foreach(policy IN ITEMS OLD NEW)
    add_test(NAME CMakeSubproject.blas_vendor_policy_${policy}
      COMMAND "${CMAKE_COMMAND}" --fresh
        -S "${CMAKE_CURRENT_SOURCE_DIR}/blas_vendor"
        -B "${CMAKE_CURRENT_BINARY_DIR}/blas_vendor_policy_${policy}"
        "-DUNI20_SOURCE_DIR=${PROJECT_SOURCE_DIR}" "-Dvendor_policy=${policy}")
  endforeach()
endif()
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
if(CMAKE_GENERATOR MATCHES "^Ninja")
  add_test(NAME CMakeSubproject.blas_vendor_multiconfig
    COMMAND "${CMAKE_COMMAND}" --fresh -G "Ninja Multi-Config"
      -S "${CMAKE_CURRENT_SOURCE_DIR}/blas_vendor/multiconfig"
      -B "${CMAKE_CURRENT_BINARY_DIR}/blas_vendor_multiconfig"
      "-DCMAKE_MAKE_PROGRAM=${CMAKE_MAKE_PROGRAM}"
      "-DUNI20_SOURCE_DIR=${PROJECT_SOURCE_DIR}")
  add_test(NAME CMakeSubproject.multiconfig
    COMMAND "${CMAKE_COMMAND}" -Dmode=subdirectory "-Dgenerator=Ninja Multi-Config"
      -Dconfig=RelWithDebInfo -P "${CMAKE_CURRENT_BINARY_DIR}/run_subproject.cmake")
  set_tests_properties(CMakeSubproject.multiconfig PROPERTIES LABELS cmake TIMEOUT 300)
endif()

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
