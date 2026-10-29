cmake_minimum_required(VERSION 3.28)
project(tbb_sanitizer_boundary LANGUAGES CXX)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
include("${UNI20_SOURCE_DIR}/cmake/Uni20TbbSanitizers.cmake")

if(case STREQUAL "enabled_semicolons" OR case STREQUAL "imported")
  set(UNI20_SANITIZE "address;undefined")
elseif(case STREQUAL "enabled_mixed")
  set(UNI20_SANITIZE "address;undefined,leak")
elseif(case STREQUAL "enabled_commas")
  set(UNI20_SANITIZE "address,undefined")
else()
  set(UNI20_SANITIZE "")
endif()
# Exercise the same normalization as the real project, including clearing a
# pre-existing list when the option is disabled.
set(UNI20_SANITIZE_LIST address)
include("${UNI20_SOURCE_DIR}/cmake/Uni20Sanitizers.cmake")
if(case STREQUAL "enabled_mixed")
  set(expected_sanitizers "address;undefined;leak")
elseif(case STREQUAL "disabled")
  set(expected_sanitizers "")
else()
  set(expected_sanitizers "address;undefined")
endif()
if(NOT UNI20_SANITIZE_LIST STREQUAL expected_sanitizers)
  message(FATAL_ERROR "Unexpected sanitizer list: ${UNI20_SANITIZE_LIST}")
endif()

if(case STREQUAL "imported")
  add_library(provider SHARED IMPORTED)
  add_library(TBB::tbb ALIAS provider)
  uni20_configure_tbb_sanitizers()
  get_target_property(options provider INTERFACE_COMPILE_OPTIONS)
  if(options)
    message(FATAL_ERROR "Modified imported provider usage requirements: ${options}")
  endif()
  return()
endif()

# The real provider is defined in a different directory, with relative source
# names. This reproduces the scope in which the old source-property fix failed.
add_compile_options(-fsanitize=address,undefined)

# Check actual probe registration without compiling or requiring MPFR in this
# configure-only fixture. The real sanitizer build compiles and runs the probe.
enable_testing()
add_library(uni20_core INTERFACE)
set(UNI20_HAS_MPREAL TRUE)
add_subdirectory("${UNI20_SOURCE_DIR}/tests/sanitizers" mpfr_policy)
get_directory_property(policy_tests DIRECTORY "${UNI20_SOURCE_DIR}/tests/sanitizers" TESTS)
# The hermetic Python symbol-preparation check does not use instrumentation
# and is intentionally available even when the sanitizer policy is disabled.
# All remaining tests here must still obey the instrumentation gate.
list(REMOVE_ITEM policy_tests "Sanitizers.PrepareSymbols")
if(CMAKE_SYSTEM_NAME STREQUAL "Linux" AND case MATCHES "^enabled_")
  if(NOT "Sanitizers.MpfrCachePolicy" IN_LIST policy_tests)
    message(FATAL_ERROR "Missing MPFR cache-policy test for ${UNI20_SANITIZE}")
  endif()
elseif(policy_tests)
  message(FATAL_ERROR "Registered sanitizer tests with the policy disabled")
endif()

file(MAKE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/provider")
file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/provider/provider.cpp" "int provider() { return 0; }\n")
file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/provider/CMakeLists.txt"
  "add_library(provider STATIC provider.cpp)\nadd_library(TBB::tbb ALIAS provider)\n")
add_subdirectory("${CMAKE_CURRENT_BINARY_DIR}/provider" provider-build)
uni20_configure_tbb_sanitizers()

file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/consumer.cpp" "int consumer() { return 0; }\n")
add_library(consumer STATIC "${CMAKE_CURRENT_BINARY_DIR}/consumer.cpp")
target_link_libraries(consumer PRIVATE TBB::tbb)
