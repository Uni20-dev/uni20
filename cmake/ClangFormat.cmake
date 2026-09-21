# ClangFormat.cmake
# This file sets up a custom target for clang-format if clang-format is available.

find_program(CLANG_FORMAT_EXE clang-format)
if(CLANG_FORMAT_EXE)
  message(STATUS "clang-format found: ${CLANG_FORMAT_EXE}")

  # List all source files that you want to format.
  file(GLOB_RECURSE ALL_CXX_SOURCE_FILES
    ${PROJECT_SOURCE_DIR}/src/*.cpp
    ${PROJECT_SOURCE_DIR}/src/*.hpp
    ${PROJECT_SOURCE_DIR}/tests/*.cpp
    ${PROJECT_SOURCE_DIR}/bindings/python/*.cpp
  )

  # Create a custom target that formats all these source files.
  set(_uni20_format_target uni20_clang_format)
  if(PROJECT_IS_TOP_LEVEL)
    set(_uni20_format_target clang_format)
  endif()
  add_custom_target(
    ${_uni20_format_target}
    COMMAND ${CLANG_FORMAT_EXE} -i ${ALL_CXX_SOURCE_FILES}
    COMMENT "Running clang-format on all source files"
  )
else()
  message(WARNING "clang-format not found. Code formatting target will be unavailable.")
endif()
