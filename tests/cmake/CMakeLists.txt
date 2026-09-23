# Configure real parent projects, then compile and run their consumers. These
# tests exercise CMake usage requirements outside Uni20's directory scope.
include(subproject_args.cmake)

if(UNI20_BUILD_CLI)
  # Exercise the public target from an embedding application and verify that
  # opting out leaves both the numerical targets and common documents CLI-free.
  get_target_property(UNI20_TEST_CLI_INCLUDE_DIRS CLI11::CLI11 INTERFACE_INCLUDE_DIRECTORIES)
  file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/cli_include_dirs.cmake"
    "set(CLI_INCLUDE_DIRS [==[${UNI20_TEST_CLI_INCLUDE_DIRS}]==])\n")
  configure_file(run_cli.cmake.in run_cli.cmake @ONLY)
  foreach(cli_case IN ITEMS disabled parent)
    add_test(NAME CMakeSubproject.cli_${cli_case}
      COMMAND "${CMAKE_COMMAND}" "-Dcli_case=${cli_case}" "-Dconfig=$<CONFIG>"
        -P "${CMAKE_CURRENT_BINARY_DIR}/run_cli.cmake")
    set_tests_properties(CMakeSubproject.cli_${cli_case} PROPERTIES LABELS cmake TIMEOUT 180)
  endforeach()
endif()

configure_file(run_initialization.cmake.in run_initialization.cmake @ONLY)
set(_uni20_initialization_cases debug debugopt release on off)
if(CMAKE_GENERATOR MATCHES "^Ninja")
  list(APPEND _uni20_initialization_cases multiconfig)
endif()
foreach(case IN LISTS _uni20_initialization_cases)
  add_test(NAME CMakeSubproject.initialization_${case}
    COMMAND "${CMAKE_COMMAND}" "-Dcase=${case}"
      -P "${CMAKE_CURRENT_BINARY_DIR}/run_initialization.cmake")
  set_tests_properties(CMakeSubproject.initialization_${case} PROPERTIES LABELS cmake TIMEOUT 60)
endforeach()

add_test(NAME CMakeSubproject.dependency_arguments
  COMMAND "${CMAKE_COMMAND}" "-DUNI20_SOURCE_DIR=${PROJECT_SOURCE_DIR}"
    "-Dtest_binary_dir=${CMAKE_CURRENT_BINARY_DIR}/dependency_arguments"
    -P "${CMAKE_CURRENT_SOURCE_DIR}/check_dependency_arguments.cmake")
set_tests_properties(CMakeSubproject.dependency_arguments PROPERTIES LABELS cmake TIMEOUT 60)

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
configure_file(run_ipo.cmake.in run_ipo.cmake @ONLY)
foreach(ipo_case IN ITEMS enabled disabled debug inherited unsupported)
  add_test(NAME CMakeSubproject.ipo_${ipo_case}
    COMMAND "${CMAKE_COMMAND}" "-Dipo_case=${ipo_case}"
      -P "${CMAKE_CURRENT_BINARY_DIR}/run_ipo.cmake")
  set_tests_properties(CMakeSubproject.ipo_${ipo_case} PROPERTIES LABELS cmake TIMEOUT 120)
endforeach()
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
  add_test(NAME CMakeSubproject.ipo_multiconfig
    COMMAND "${CMAKE_COMMAND}" -Dipo_case=enabled "-Dgenerator=Ninja Multi-Config"
      -P "${CMAKE_CURRENT_BINARY_DIR}/run_ipo.cmake")
  set_tests_properties(CMakeSubproject.ipo_multiconfig PROPERTIES LABELS cmake TIMEOUT 120)
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
