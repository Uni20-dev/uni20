cmake_minimum_required(VERSION 3.24)
project(uni20_consumer LANGUAGES CXX)

if(NOT UNI20_SOURCE_DIR)
  message(FATAL_ERROR "Pass -DUNI20_SOURCE_DIR=/path/to/uni20")
endif()
set(UNI20_INTEGRATION_MODE "subdirectory" CACHE STRING "subdirectory or fetchcontent")
option(UNI20_TEST_DEVELOPER_TARGETS "Opt into embedded documentation and formatting" OFF)
option(UNI20_TEST_PYTHON "Build and run embedded Python tests" OFF)

# A consumer may use an older language standard for its unrelated targets.
# Uni20 must advertise its own requirement through the linked library.
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(consumer_configurations Debug Release RelWithDebInfo MinSizeRel DebugOpt ConsumerSpecial)
if(CMAKE_BUILD_TYPE)
  list(APPEND consumer_configurations "${CMAKE_BUILD_TYPE}")
endif()
list(REMOVE_DUPLICATES consumer_configurations)
set(CMAKE_CONFIGURATION_TYPES "${consumer_configurations}" CACHE STRING "Parent configurations" FORCE)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(FETCHCONTENT_BASE_DIR "${CMAKE_CURRENT_BINARY_DIR}/parent-deps" CACHE PATH "Parent dependencies" FORCE)
set(FMT_TEST OFF CACHE BOOL "Parent fmt setting" FORCE)
set(BUILD_GMOCK ON CACHE BOOL "Parent gmock setting" FORCE)
set(BLA_VENDOR "Generic" CACHE STRING "Parent BLAS setting" FORCE)
set(BLA_SIZEOF_INTEGER 8 CACHE STRING "Parent BLAS integer width" FORCE)
set(TBB_TEST ON CACHE BOOL "Parent oneTBB setting" FORCE)
if(UNI20_TEST_PARENT_BLAS STREQUAL "matching")
  set(BLA_SIZEOF_INTEGER 4 CACHE STRING "Parent BLAS integer width" FORCE)
  find_package(BLAS REQUIRED)
  find_package(LAPACK REQUIRED)
  get_target_property(parent_blas_libraries BLAS::BLAS INTERFACE_LINK_LIBRARIES)
  get_target_property(parent_lapack_libraries LAPACK::LAPACK INTERFACE_LINK_LIBRARIES)
elseif(UNI20_TEST_PARENT_BLAS STREQUAL "conflicting")
  # The ABI guard must reject this before discovery or linking. No ILP64
  # installation is necessary to exercise the configuration boundary.
  add_library(BLAS::BLAS INTERFACE IMPORTED)
elseif(UNI20_TEST_PARENT_BLAS MATCHES "^imported")
  # No find_package-created targets or package state: even a machine with
  # installed providers must reject any attempt to rediscover these targets.
  set(BLA_SIZEOF_INTEGER 4 CACHE STRING "Parent BLAS integer width" FORCE)
  add_library(BLAS::BLAS INTERFACE IMPORTED)
  add_library(LAPACK::LAPACK INTERFACE IMPORTED)
  set(CMAKE_DISABLE_FIND_PACKAGE_BLAS TRUE)
  set(CMAKE_DISABLE_FIND_PACKAGE_LAPACK TRUE)
  set(BLAS_LIBRARIES "/unrelated/libmkl_rt.so")
  set(BLAS_LINKER_FLAGS "-stale-blas-link-option")
  set(LAPACK_LINKER_FLAGS "-stale-lapack-link-option")
  if(UNI20_TEST_PARENT_BLAS STREQUAL "imported_no_blas")
    set(UNI20_BACKEND_BLAS OFF CACHE BOOL "Exercise LAPACK without the BLAS backend" FORCE)
  endif()
endif()

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
  # Record the generated commands without formatting the real source checkout
  # or generating the full API documentation during a CMake regression.
  find_package(Python3 REQUIRED COMPONENTS Interpreter)
  configure_file(record_tool.py.in record_tool.py @ONLY)
  file(CHMOD "${CMAKE_CURRENT_BINARY_DIR}/record_tool.py"
    PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE)
  set(CLANG_FORMAT_EXE "${CMAKE_CURRENT_BINARY_DIR}/record_tool.py" CACHE FILEPATH "Test recorder" FORCE)
  file(MAKE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/modules")
  file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/modules/FindDoxygen.cmake"
    "set(DOXYGEN_FOUND TRUE)\nset(DOXYGEN_EXECUTABLE \"${CMAKE_CURRENT_BINARY_DIR}/record_tool.py\")\n")
  list(PREPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_BINARY_DIR}/modules")
endif()
if(UNI20_TEST_PYTHON)
  set(UNI20_BUILD_TESTS ON CACHE BOOL "" FORCE)
  set(UNI20_BUILD_PYTHON ON CACHE BOOL "" FORCE)
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

# A dependency may publish several build includes inside one generator
# expression (as MPLAPACK does). Warning policy must preserve its exports.
file(MAKE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/export-probe")
file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/export-probe/CMakeLists.txt" [=[
add_library(consumer_export_probe INTERFACE)
target_include_directories(consumer_export_probe INTERFACE
  "$<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include-one;${CMAKE_CURRENT_BINARY_DIR}/include-two>"
  "$<INSTALL_INTERFACE:include>")
relax_warnings(consumer_export_probe)
install(TARGETS consumer_export_probe EXPORT consumer_export_probe)
install(EXPORT consumer_export_probe DESTINATION lib/cmake/consumer_export_probe)
]=])
add_subdirectory("${CMAKE_CURRENT_BINARY_DIR}/export-probe"
  "${CMAKE_CURRENT_BINARY_DIR}/export-probe-build")

if(UNI20_TEST_PARENT_BLAS MATCHES "^imported")
  get_target_property(lapack_options uni20_backend_lapack INTERFACE_LINK_OPTIONS)
  if(lapack_options)
    message(FATAL_ERROR "Target-owned LAPACK inherited stale package link options")
  endif()
  if(UNI20_BACKEND_BLAS)
    get_target_property(blas_options uni20_interface_blas INTERFACE_LINK_OPTIONS)
    if(blas_options OR NOT UNI20_DETECTED_BLAS_VENDOR STREQUAL "Generic")
      message(FATAL_ERROR "Target-owned BLAS inherited unrelated package state")
    endif()
  endif()
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
    UNI20_BUILD_BENCH UNI20_BUILD_EXAMPLES UNI20_BUILD_PYTHON UNI20_BUILD_ASM UNI20_ENABLE_LTO)
  if(UNI20_TEST_PYTHON AND (option STREQUAL "UNI20_BUILD_TESTS" OR option STREQUAL "UNI20_BUILD_PYTHON"))
    continue()
  endif()
  if(${option})
    message(FATAL_ERROR "${option} must default to OFF when embedded")
  endif()
endforeach()
if(NOT UNI20_TEST_DEVELOPER_TARGETS AND (UNI20_BUILD_DOCS OR UNI20_BUILD_FORMAT_TARGET))
  message(FATAL_ERROR "Developer targets must default to OFF when embedded")
endif()

if(UNI20_TEST_PYTHON AND UNI20_GTEST_SOURCE STREQUAL "fetched" AND NOT TARGET GTest::gmock)
  message(FATAL_ERROR "Parent BUILD_GMOCK=ON must create the requested component")
endif()
if(UNI20_TBB_SOURCE STREQUAL "fetched" AND NOT TARGET test_tick_count)
  message(FATAL_ERROR "Parent TBB_TEST=ON must create oneTBB test targets")
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
if(UNI20_TEST_PARENT_BLAS STREQUAL "matching")
  get_target_property(current_blas_libraries BLAS::BLAS INTERFACE_LINK_LIBRARIES)
  get_target_property(current_lapack_libraries LAPACK::LAPACK INTERFACE_LINK_LIBRARIES)
  if(NOT current_blas_libraries STREQUAL parent_blas_libraries OR
      NOT current_lapack_libraries STREQUAL parent_lapack_libraries)
    message(FATAL_ERROR "Uni20 changed the parent's numerical provider targets")
  endif()
  target_link_libraries(uni20_consumer PRIVATE uni20_backend_blas_reference uni20_backend_lapack)
  target_compile_definitions(uni20_consumer PRIVATE UNI20_TEST_PARENT_BLAS=1)
endif()
enable_testing()
add_test(NAME consumer_runs COMMAND uni20_consumer)
