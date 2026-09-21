include(FetchContent)
set(UNI20_SUBPROJECT_TEST_ARGS "")
foreach(dependency IN ITEMS fmt mdspan TBB GTest nanobind)
  unset(dependency_source)
  FetchContent_GetProperties(${dependency} SOURCE_DIR dependency_source)
  if(dependency_source)
    string(TOUPPER "${dependency}" dependency_upper)
    string(APPEND UNI20_SUBPROJECT_TEST_ARGS
      "  [==[-DFETCHCONTENT_SOURCE_DIR_${dependency_upper}=${dependency_source}]==]\n")
  endif()
endforeach()

foreach(variable IN ITEMS CMAKE_CXX_COMPILER CMAKE_C_COMPILER CMAKE_TOOLCHAIN_FILE
    CMAKE_GENERATOR_PLATFORM CMAKE_GENERATOR_TOOLSET CMAKE_GENERATOR_INSTANCE
    CMAKE_MAKE_PROGRAM CMAKE_PREFIX_PATH fmt_DIR TBB_DIR mplapack_DIR
    GTest_DIR nanobind_DIR UNI20_USE_SYSTEM_GTEST UNI20_USE_SYSTEM_NANOBIND
    FETCHCONTENT_FULLY_DISCONNECTED UNI20_ENABLE_STACKTRACE)
  if(DEFINED ${variable} AND NOT "${${variable}}" MATCHES "-NOTFOUND$")
    string(APPEND UNI20_SUBPROJECT_TEST_ARGS "  [==[-D${variable}=${${variable}}]==]\n")
  endif()
endforeach()
