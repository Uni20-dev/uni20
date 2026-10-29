file(WRITE "${test_binary_dir}/metadata-only.supp" "leak:uni20/async/async_node.hpp\n")
file(WRITE "${test_binary_dir}/unanchored.supp"
  "leak:uni20/async/async_node.hpp\nleak:mpfr_atan\n")

function(check mode policy expected)
  execute_process(COMMAND "${CMAKE_COMMAND}" -E env
    "DEBUGINFOD_URLS=" "ASAN_OPTIONS=detect_leaks=1:halt_on_error=1"
    "UBSAN_OPTIONS=halt_on_error=1:print_stacktrace=1"
    "LSAN_OPTIONS=suppressions=${policy}:print_suppressions=0:exitcode=23:use_tls=0"
    "${probe}" "${mode}"
    RESULT_VARIABLE result OUTPUT_VARIABLE output ERROR_VARIABLE error TIMEOUT 15)
  if(NOT "${result}" STREQUAL "${expected}")
    message(FATAL_ERROR "${mode} with ${policy}: expected ${expected}, got ${result}\n${output}\n${error}")
  endif()
  if(expected EQUAL 23 AND NOT error MATCHES "LeakSanitizer: detected memory leaks")
    message(FATAL_ERROR "${mode}: failed without the expected leak report\n${output}\n${error}")
  endif()
endfunction()

# Compare the same arithmetic with and without local cache cleanup. Owned
# values have already been destroyed when cleanup runs. Ignoring TLS roots in
# this isolated probe exposes caches without relying on worker termination.
# This option is not part of the recommended application sanitizer environment.
check(cache "${test_binary_dir}/metadata-only.supp" 23)
check(clean "${test_binary_dir}/metadata-only.supp" 0)
check(cache "${suppressions}" 0)
check(owned "${suppressions}" 23)
check(plain "${suppressions}" 23)
# This deliberately unsafe rule demonstrates why exact C-symbol anchors matter.
check(owned "${test_binary_dir}/unanchored.supp" 0)

# The exponential's internal mpfr_exp_2 path uses a separate per-thread pool.
# Verify its local cleanup and retain leak detection for the owning exp result.
check(cache-exp "${test_binary_dir}/metadata-only.supp" 23)
check(clean-exp "${test_binary_dir}/metadata-only.supp" 0)
check(cache-exp "${suppressions}" 0)
check(owned-exp "${suppressions}" 23)

# Log-Gamma uses a Bernoulli-number cache, distinct from the constant and
# exponential pools. Local cleanup must release it; owning results still leak.
check(cache-lgamma "${test_binary_dir}/metadata-only.supp" 23)
check(clean-lgamma "${test_binary_dir}/metadata-only.supp" 0)
check(cache-lgamma "${suppressions}" 0)
check(owned-lgamma "${suppressions}" 23)
