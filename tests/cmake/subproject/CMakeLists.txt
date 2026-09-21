cmake_minimum_required(VERSION 3.24)
project(uni20_consumer LANGUAGES CXX)

if(NOT UNI20_SOURCE_DIR)
  message(FATAL_ERROR "Pass -DUNI20_SOURCE_DIR=/path/to/uni20")
endif()
set(UNI20_INTEGRATION_MODE "subdirectory" CACHE STRING "subdirectory or fetchcontent")
option(UNI20_TEST_DEVELOPER_TARGETS "Opt into embedded documentation and formatting" OFF)

# A consumer may use an older language standard for its unrelated targets.
# Uni20 must advertise its own requirement through the linked library.
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CONFIGURATION_TYPES "Debug;Release;ConsumerSpecial" CACHE STRING "Parent configurations" FORCE)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(FETCHCONTENT_BASE_DIR "${CMAKE_CURRENT_BINARY_DIR}/parent-deps" CACHE PATH "Parent dependencies" FORCE)
set(FMT_TEST OFF CACHE BOOL "Parent fmt setting" FORCE)
set(BUILD_GMOCK ON CACHE BOOL "Parent gmock setting" FORCE)
set(BLA_VENDOR "Generic" CACHE STRING "Parent BLAS setting" FORCE)
set(BLA_SIZEOF_INTEGER 8 CACHE STRING "Parent BLAS integer width" FORCE)
set(TBB_TEST ON CACHE BOOL "Parent oneTBB setting" FORCE)

foreach(variable IN ITEMS CMAKE_BUILD_TYPE CMAKE_CONFIGURATION_TYPES
    CMAKE_CXX_FLAGS_DEBUG CMAKE_CXX_FLAGS_RELEASE CMAKE_CXX_STANDARD
    CMAKE_POSITION_INDEPENDENT_CODE BUILD_GMOCK BLA_VENDOR BLA_SIZEOF_INTEGER
    TBB_TEST FETCHCONTENT_BASE_DIR)
  set("saved_${variable}" "${${variable}}")
endforeach()

# These common target names belong to the parent project.
add_custom_target(doc)
add_custom_target(clang_format)
add_custom_target(coverage)

if(UNI20_TEST_DEVELOPER_TARGETS)
  set(UNI20_BUILD_DOCS ON CACHE BOOL "" FORCE)
  set(UNI20_BUILD_FORMAT_TARGET ON CACHE BOOL "" FORCE)
endif()

if(UNI20_INTEGRATION_MODE STREQUAL "fetchcontent")
  include(FetchContent)
  set(FETCHCONTENT_SOURCE_DIR_UNI20 "${UNI20_SOURCE_DIR}")
  FetchContent_Declare(uni20 SOURCE_DIR "${UNI20_SOURCE_DIR}")
  FetchContent_MakeAvailable(uni20)
  set(embedded_binary_dir "${uni20_BINARY_DIR}")
elseif(UNI20_INTEGRATION_MODE STREQUAL "subdirectory")
  set(embedded_binary_dir "${CMAKE_CURRENT_BINARY_DIR}/uni20")
  add_subdirectory("${UNI20_SOURCE_DIR}" "${embedded_binary_dir}")
else()
  message(FATAL_ERROR "Unknown integration mode: ${UNI20_INTEGRATION_MODE}")
endif()

foreach(variable IN ITEMS CMAKE_BUILD_TYPE CMAKE_CONFIGURATION_TYPES
    CMAKE_CXX_FLAGS_DEBUG CMAKE_CXX_FLAGS_RELEASE CMAKE_CXX_STANDARD
    CMAKE_POSITION_INDEPENDENT_CODE BUILD_GMOCK BLA_VENDOR BLA_SIZEOF_INTEGER
    TBB_TEST FETCHCONTENT_BASE_DIR)
  if(NOT "${${variable}}" STREQUAL "${saved_${variable}}")
    message(FATAL_ERROR "Uni20 changed parent setting ${variable}")
  endif()
endforeach()

foreach(option IN ITEMS UNI20_BUILD_TESTS UNI20_BUILD_COMBINED_TESTS
    UNI20_BUILD_BENCH UNI20_BUILD_EXAMPLES UNI20_BUILD_PYTHON UNI20_BUILD_ASM)
  if(${option})
    message(FATAL_ERROR "${option} must default to OFF when embedded")
  endif()
endforeach()
if(NOT UNI20_TEST_DEVELOPER_TARGETS AND (UNI20_BUILD_DOCS OR UNI20_BUILD_FORMAT_TARGET))
  message(FATAL_ERROR "Developer targets must default to OFF when embedded")
endif()
if(UNI20_FETCHCONTENT_SOURCE)
  message(FATAL_ERROR "Embedded dependency sources should default to build-local storage")
endif()
if(NOT UNI20_FETCHCONTENT_BASE_DIR STREQUAL "${embedded_binary_dir}/.cmake/third_party")
  message(FATAL_ERROR "Uni20 dependency builds escaped its binary directory")
endif()

if(UNI20_TEST_DEVELOPER_TARGETS)
  if(NOT TARGET uni20_doc OR NOT TARGET uni20_clang_format)
    message(FATAL_ERROR "This check requires Doxygen and clang-format")
  endif()
  file(READ "${embedded_binary_dir}/docs/Doxyfile" doxyfile)
  string(FIND "${doxyfile}" "${UNI20_SOURCE_DIR}/docs/mainpage.md" mainpage)
  if(mainpage EQUAL -1)
    message(FATAL_ERROR "Doxygen uses the consumer's source tree")
  endif()
endif()

add_executable(uni20_consumer main.cpp)
target_link_libraries(uni20_consumer PRIVATE uni20_core uni20_common)
enable_testing()
add_test(NAME consumer_runs COMMAND uni20_consumer)
