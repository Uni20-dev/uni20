cmake_minimum_required(VERSION 3.28)
project(package_hint_contract LANGUAGES NONE)

# Packages live outside every default search prefix, with no fetched sources.
# The child must receive the directory hints to resolve these marked packages.
foreach(package IN ITEMS GTest nanobind)
  set(package_dir "${CMAKE_CURRENT_BINARY_DIR}/private packages/${package}")
  file(MAKE_DIRECTORY "${package_dir}")
  file(WRITE "${package_dir}/${package}Config.cmake"
    "add_library(${package}::fixture INTERFACE IMPORTED)\n")
  set(${package}_DIR "${package_dir}")
endforeach()
set(UNI20_USE_SYSTEM_GTEST ON)
set(UNI20_USE_SYSTEM_NANOBIND ON)
set(FETCHCONTENT_FULLY_DISCONNECTED ON)
include("${UNI20_SOURCE_DIR}/tests/cmake/subproject_args.cmake")
configure_file(run_child.cmake.in run_child.cmake @ONLY)
execute_process(COMMAND "${CMAKE_COMMAND}" -P "${CMAKE_CURRENT_BINARY_DIR}/run_child.cmake"
  RESULT_VARIABLE result OUTPUT_VARIABLE output ERROR_VARIABLE error)
if(NOT result EQUAL 0)
  message(FATAL_ERROR "Package hints did not reach the child: ${output}\n${error}")
endif()
