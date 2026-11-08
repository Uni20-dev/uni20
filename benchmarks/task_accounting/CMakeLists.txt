# The same baseline source is built with and without the accounting definitions.
# These switches affect only the measurement executables, never library headers.
foreach(enabled IN ITEMS 0 1)
  add_executable(uni20_accounting_bench_${enabled} benchmark.cpp)
  target_link_libraries(uni20_accounting_bench_${enabled} PRIVATE uni20_async TBB::tbb)
  target_compile_definitions(uni20_accounting_bench_${enabled} PRIVATE UNI20_ACCOUNTING_PROBE=${enabled})
  add_library(uni20_accounting_codegen_${enabled} OBJECT codegen.cpp)
  target_link_libraries(uni20_accounting_codegen_${enabled} PRIVATE uni20_async)
  target_compile_definitions(uni20_accounting_codegen_${enabled} PRIVATE UNI20_ACCOUNTING_PROBE=${enabled})
  add_executable(uni20_accounting_allocations_${enabled} allocations.cpp)
  target_link_libraries(uni20_accounting_allocations_${enabled} PRIVATE uni20_async)
  target_compile_definitions(uni20_accounting_allocations_${enabled} PRIVATE UNI20_ACCOUNTING_PROBE=${enabled})
endforeach()
