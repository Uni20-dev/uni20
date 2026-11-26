# Configure small parents that create a dependency target before Uni20 resolves
# it. Uni20 links such a target as it is, so it must reject a known version below
# its minimum, even when its own package hint names a newer installation, and
# report the version it actually links.
file(REMOVE_RECURSE "${test_binary_dir}")
set(packages "${test_binary_dir}/packages")

# Installed Fixture packages. Like CMake's export files, each defines its target
# only once, so a later lookup reuses the target that already exists.
include(CMakePackageConfigHelpers)
foreach(version IN ITEMS 1.0 2.5 3.0)
  file(WRITE "${packages}/${version}/FixtureConfig.cmake" [=[
if(NOT TARGET Fixture::library)
  add_library(Fixture::library INTERFACE IMPORTED)
endif()
]=])
  write_basic_package_version_file("${packages}/${version}/FixtureConfigVersion.cmake"
    VERSION ${version} COMPATIBILITY SameMajorVersion)
endforeach()

set(parent "${test_binary_dir}/parent")
file(WRITE "${parent}/CMakeLists.txt" [=[
cmake_minimum_required(VERSION 3.28)
project(existing_target LANGUAGES NONE)
if(case MATCHES "^cli")
  add_library(CLI11::CLI11 INTERFACE IMPORTED)
  set(CLI11_VERSION ${parent_version})
elseif(parent_version)
  find_package(Fixture ${parent_version} EXACT CONFIG REQUIRED
               NO_DEFAULT_PATH PATHS "${packages}/${parent_version}")
else()
  add_library(Fixture::library INTERFACE IMPORTED)
endif()
add_subdirectory(child)
]=])
file(WRITE "${parent}/child/CMakeLists.txt" [=[
include("${UNI20_SOURCE_DIR}/cmake/Uni20Dependencies.cmake")
if(case MATCHES "^cli")
  include("${UNI20_SOURCE_DIR}/cmake/Uni20CLI.cmake")
  return()
endif()
if(child_version)
  set(Fixture_DIR "${packages}/${child_version}")
endif()
set(UNI20_USE_SYSTEM_FIXTURE ${mode})
uni20_add_dependency(NAME Fixture VERSION 2.0 TARGET Fixture::library
                     REPO "${packages}/unused" TAG unused)
message(STATUS "Fixture resolved as ${UNI20_FIXTURE_SOURCE} ${UNI20_FIXTURE_VERSION}")
]=])

# case | mode | parent's version | child's package hint | expected output
set(cases
  "older|AUTO|1.0||Uni20 requires Fixture >= 2.0, but target 'Fixture::library' already exists with Fixture 1.0"
  "older_child_hint|ON|1.0|2.5|Uni20 requires Fixture >= 2.0, but target 'Fixture::library' already exists with Fixture 1.0"
  "newer|ON|3.0||Fixture resolved as system 3.0"
  "newer_child_hint|ON|3.0|2.5|Fixture resolved as system 3.0"
  "unversioned|ON|||Fixture resolved as system"
  "cli_older|ON|2.6.0||Uni20 requires CLI11 >= 2.7.2, but target 'CLI11::CLI11' already exists with CLI11 2.6.0"
  "cli_newer|ON|2.8.0||")
foreach(entry IN LISTS cases)
  string(REPLACE "|" ";" fields "${entry}")
  list(GET fields 0 case)
  list(GET fields 1 mode)
  list(GET fields 2 parent_version)
  list(GET fields 3 child_version)
  list(GET fields 4 expected)
  execute_process(
    COMMAND "${CMAKE_COMMAND}" --fresh -S "${parent}" -B "${test_binary_dir}/build/${case}"
      "-DUNI20_SOURCE_DIR=${UNI20_SOURCE_DIR}" "-Dpackages=${packages}" "-Dcase=${case}"
      "-Dmode=${mode}" "-Dparent_version=${parent_version}" "-Dchild_version=${child_version}"
    RESULT_VARIABLE result OUTPUT_VARIABLE output ERROR_VARIABLE error TIMEOUT 30)
  string(REGEX REPLACE "[ \t\r\n]+" " " messages "${output} ${error}")
  string(FIND "${messages}" "${expected}" expected_offset)
  set(failed TRUE)
  if(result EQUAL 0)
    set(failed FALSE)
  endif()
  set(should_fail FALSE)
  if(case MATCHES "older")
    set(should_fail TRUE)
  endif()
  if(NOT failed STREQUAL should_fail OR expected_offset EQUAL -1)
    message(FATAL_ERROR
      "Case ${case}: expected ${should_fail} failure with '${expected}', got result ${result}:\n"
      "${output}\n${error}")
  endif()
endforeach()
