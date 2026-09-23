# tests/common/CMakeLists.txt

add_test_module(common
  SOURCES test_terminal_color.cpp test_trace.cpp test_trace_debug.cpp test_trace_ndebug.cpp
          test_aligned_buffer.cpp test_floating_eq.cpp test_namedenum.cpp test_string_util.cpp
          test_trace_format.cpp test_terminal_utils.cpp test_presentation.cpp test_display.cpp test_nifty_counter.cpp
          test_process.cpp test_performance_measurements.cpp test_initialization.cpp test_half_int.cpp test_data_table.cpp test_data_table_streaming.cpp test_data_table_json.cpp
  LIBS uni20_common mdspan
)

add_executable(uni20_initialization_diagnostic_probe initialization_diagnostic_probe.cpp)
target_link_libraries(uni20_initialization_diagnostic_probe PRIVATE uni20_deps)
add_test(NAME InitializationDiagnostic.native_complete
  COMMAND uni20_initialization_diagnostic_probe complete)

set(_uni20_initialization_tools "")
# MSan and Memcheck each own the process's address-space instrumentation; run
# their test configurations separately even when client requests are compiled in.
if(UNI20_ENABLE_VALGRIND AND NOT "memory" IN_LIST UNI20_SANITIZE_LIST)
  find_program(UNI20_VALGRIND_EXECUTABLE valgrind)
  mark_as_advanced(UNI20_VALGRIND_EXECUTABLE)
  if(UNI20_VALGRIND_EXECUTABLE)
    list(APPEND _uni20_initialization_tools valgrind)
  endif()
endif()
if("memory" IN_LIST UNI20_SANITIZE_LIST)
  list(APPEND _uni20_initialization_tools msan)
endif()
foreach(tool IN LISTS _uni20_initialization_tools)
  foreach(mode IN ITEMS complete missing copy)
    add_test(NAME InitializationDiagnostic.${tool}_${mode}
      COMMAND "${CMAKE_COMMAND}" "-Dtool=${tool}" "-Dmode=${mode}"
        "-Dprobe=$<TARGET_FILE:uni20_initialization_diagnostic_probe>"
        "-Dvalgrind=${UNI20_VALGRIND_EXECUTABLE}"
        -P "${CMAKE_CURRENT_SOURCE_DIR}/run_initialization_diagnostic.cmake")
    set_tests_properties(InitializationDiagnostic.${tool}_${mode} PROPERTIES LABELS initialization TIMEOUT 60)
  endforeach()
endforeach()
