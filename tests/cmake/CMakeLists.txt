# Configure real parent projects, then compile and run their consumers. These
# tests exercise CMake usage requirements outside Uni20's directory scope.
include(FetchContent)
set(UNI20_SUBPROJECT_TEST_ARGS "")
foreach(dependency IN ITEMS fmt mdspan TBB)
  unset(dependency_source)
  FetchContent_GetProperties(${dependency} SOURCE_DIR dependency_source)
  if(dependency_source)
    string(TOUPPER "${dependency}" dependency_upper)
    string(APPEND UNI20_SUBPROJECT_TEST_ARGS
      "  [==[-DFETCHCONTENT_SOURCE_DIR_${dependency_upper}=${dependency_source}]==]\n")
  endif()
endforeach()

foreach(variable IN ITEMS CMAKE_CXX_COMPILER CMAKE_C_COMPILER CMAKE_TOOLCHAIN_FILE
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
foreach(mode IN ITEMS subdirectory fetchcontent)
  add_test(NAME CMakeSubproject.${mode}
    COMMAND "${CMAKE_COMMAND}" "-Dmode=${mode}" "-Dconfig=$<CONFIG>"
      -P "${CMAKE_CURRENT_BINARY_DIR}/run_subproject.cmake")
  set_tests_properties(CMakeSubproject.${mode} PROPERTIES LABELS cmake TIMEOUT 300)
endforeach()

find_package(Doxygen QUIET)
find_program(UNI20_TEST_CLANG_FORMAT clang-format)
if(DOXYGEN_FOUND AND UNI20_TEST_CLANG_FORMAT)
  add_test(NAME CMakeSubproject.developer_targets
    COMMAND "${CMAKE_COMMAND}" -Dmode=subdirectory -Ddeveloper_targets=ON
      "-Dconfig=$<CONFIG>" -P "${CMAKE_CURRENT_BINARY_DIR}/run_subproject.cmake")
  set_tests_properties(CMakeSubproject.developer_targets PROPERTIES LABELS cmake TIMEOUT 300)
endif()
