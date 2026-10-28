file(WRITE "${test_binary_dir}/dag-no-suppressions.supp" "")

function(check mode policy expected)
  execute_process(COMMAND "${CMAKE_COMMAND}" -E env
    "DEBUGINFOD_URLS=" "ASAN_OPTIONS=detect_leaks=1:halt_on_error=1"
    "UBSAN_OPTIONS=halt_on_error=1:print_stacktrace=1"
    "LSAN_OPTIONS=suppressions=${policy}:print_suppressions=0:exitcode=23"
    "${probe}" "${mode}"
    RESULT_VARIABLE result OUTPUT_VARIABLE output ERROR_VARIABLE error TIMEOUT 15)
  if(NOT "${result}" STREQUAL "${expected}")
    message(FATAL_ERROR "${mode} with ${policy}: expected ${expected}, got ${result}\n${output}\n${error}")
  endif()
  if(expected EQUAL 23 AND NOT error MATCHES "LeakSanitizer: detected memory leaks")
    message(FATAL_ERROR "${mode}: failed without the expected leak report\n${output}\n${error}")
  endif()
endfunction()

check(clean "${test_binary_dir}/dag-no-suppressions.supp" 0)
check(metadata-raw "${test_binary_dir}/dag-no-suppressions.supp" 23)
check(metadata-storage "${test_binary_dir}/dag-no-suppressions.supp" 23)
check(metadata-raw "${suppressions}" 0)
check(metadata-storage "${suppressions}" 0)
check(owned "${suppressions}" 23)
check(plain "${suppressions}" 23)
