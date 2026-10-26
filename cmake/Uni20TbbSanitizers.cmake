# Keep the historical oneTBB instrumentation boundary local to its compiled
# implementation. Public headers instantiated by consumers remain instrumented.
function(uni20_configure_tbb_sanitizers)
  if(NOT UNI20_SANITIZE MATCHES "(^|,)(address|undefined|thread)(,|$)")
    return()
  endif()

  get_target_property(_tbb_target TBB::tbb ALIASED_TARGET)
  if(NOT _tbb_target)
    set(_tbb_target TBB::tbb)
  endif()
  get_target_property(_tbb_imported "${_tbb_target}" IMPORTED)
  if(_tbb_imported)
    message(STATUS "oneTBB is prebuilt; its sanitizer instrumentation is unchanged")
    return()
  endif()

  message(STATUS "Disabling sanitizer instrumentation for the oneTBB implementation (${UNI20_SANITIZE})")
  # Source properties set here would not reach oneTBB's directory. PRIVATE
  # target options apply there without disabling checks in linked consumers.
  target_compile_options("${_tbb_target}" PRIVATE -fno-sanitize=all)
  if(UNI20_SANITIZE MATCHES "(^|,)(address|thread)(,|$)")
    # Disabling instrumentation also disables the compiler's sanitizer macros.
    # Preserve oneTBB's sanitizer-selected coroutine implementation: switching
    # an instrumented caller through makecontext/swapcontext is unsupported.
    target_compile_definitions("${_tbb_target}" PRIVATE __TBB_RESUMABLE_TASKS_USE_THREADS=1)
  endif()
endfunction()
