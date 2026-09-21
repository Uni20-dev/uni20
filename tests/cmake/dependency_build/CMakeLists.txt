cmake_minimum_required(VERSION 3.28)
project(dependency_build LANGUAGES CXX)
include("${UNI20_SOURCE_DIR}/cmake/Uni20Dependencies.cmake")

if(CMAKE_CONFIGURATION_TYPES AND CMAKE_BUILD_TYPE)
  list(APPEND CMAKE_CONFIGURATION_TYPES "${CMAKE_BUILD_TYPE}")
  list(REMOVE_DUPLICATES CMAKE_CONFIGURATION_TYPES)
endif()

set(UNI20_FETCHCONTENT_BASE_DIR "${CMAKE_CURRENT_BINARY_DIR}/dependencies")
set(UNI20_FETCHCONTENT_SOURCE_BASE_DIR "${UNI20_FETCHCONTENT_BASE_DIR}")
set(FETCHCONTENT_BASE_DIR "${UNI20_FETCHCONTENT_BASE_DIR}")
set(FETCHCONTENT_SOURCE_DIR_FIXTURE "${CMAKE_CURRENT_BINARY_DIR}/fixture-source")
set(UNI20_USE_SYSTEM_FIXTURE OFF)
file(MAKE_DIRECTORY "${FETCHCONTENT_SOURCE_DIR_FIXTURE}/include")
file(WRITE "${FETCHCONTENT_SOURCE_DIR_FIXTURE}/CMakeLists.txt" [=[
cmake_minimum_required(VERSION 3.28)
project(Fixture LANGUAGES CXX)
add_library(fixture STATIC library.cpp)
target_include_directories(fixture PUBLIC
  "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
  "$<INSTALL_INTERFACE:include>")
add_custom_target(dependency_extra ALL
  COMMAND "${CMAKE_COMMAND}" -E touch "${CMAKE_CURRENT_BINARY_DIR}/extra-built")
]=])
file(WRITE "${FETCHCONTENT_SOURCE_DIR_FIXTURE}/include/dependency.hpp" [=[
// An ordinary include triggers -Wunused-parameter in the consuming target.
inline int dependency_header(int unused) { return 42; }
int dependency_value();
]=])
file(WRITE "${FETCHCONTENT_SOURCE_DIR_FIXTURE}/library.cpp" [=[
#include <dependency.hpp>
int dependency_value() { return dependency_header(0); }
]=])
uni20_add_dependency(NAME Fixture TARGET fixture)

file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/main.cpp" [=[
#include <dependency.hpp>
int main() { return dependency_value() == 42 ? 0 : 1; }
]=])
add_executable(dependency_consumer "${CMAKE_CURRENT_BINARY_DIR}/main.cpp")
target_link_libraries(dependency_consumer PRIVATE fixture)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  target_compile_options(dependency_consumer PRIVATE -Wall -Wextra -Werror)
endif()
enable_testing()
add_test(NAME dependency_consumer_runs COMMAND dependency_consumer)
