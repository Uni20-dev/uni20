# Installed packages only: MPFR/GMP use an independent configure/make build.
include_guard(GLOBAL)

function(uni20_find_mpfr)
  find_path(UNI20_GMP_INCLUDE_DIR NAMES gmp.h)
  find_library(UNI20_GMP_LIBRARY NAMES gmp)
  find_path(UNI20_MPFR_INCLUDE_DIR NAMES mpfr.h)
  find_library(UNI20_MPFR_LIBRARY NAMES mpfr)
  if(NOT UNI20_GMP_INCLUDE_DIR OR NOT UNI20_GMP_LIBRARY OR
     NOT UNI20_MPFR_INCLUDE_DIR OR NOT UNI20_MPFR_LIBRARY)
    message(FATAL_ERROR
      "UNI20_ENABLE_MPFR requires installed GMP and MPFR development packages. "
      "On Ubuntu: sudo apt install libgmp-dev libmpfr-dev. "
      "For a custom installation, set CMAKE_PREFIX_PATH or UNI20_{GMP,MPFR}_{INCLUDE_DIR,LIBRARY}. "
      "Uni20 does not download these libraries.")
  endif()

  include(CheckCXXSourceCompiles)
  include(CheckCXXSourceRuns)
  set(CMAKE_REQUIRED_INCLUDES "${UNI20_MPFR_INCLUDE_DIR};${UNI20_GMP_INCLUDE_DIR}")
  set(CMAKE_REQUIRED_LIBRARIES "${UNI20_MPFR_LIBRARY};${UNI20_GMP_LIBRARY}")
  # Recheck when a user changes package paths in an existing build tree.
  unset(UNI20_MPFR_LINKS CACHE)
  set(CMAKE_TRY_COMPILE_TARGET_TYPE EXECUTABLE)
  check_cxx_source_compiles([=[
    #include <mpfr.h>
    #if MPFR_VERSION < MPFR_VERSION_NUM(4, 1, 0)
    #error Uni20 requires MPFR 4.1 or newer
    #endif
    int main() {
      mpq_t q; mpq_init(q); mpq_set_ui(q, 1, 3);
      mpfr_t x; mpfr_init2(x, 256); mpfr_set_q(x, q, MPFR_RNDN);
      mpfr_clear(x); mpq_clear(q);
      return 0;
    }
  ]=] UNI20_MPFR_LINKS)
  if(NOT UNI20_MPFR_LINKS)
    message(FATAL_ERROR "Cannot compile and link MPFR >= 4.1 with GMP; check the selected headers and libraries.")
  endif()
  if(NOT CMAKE_CROSSCOMPILING OR CMAKE_CROSSCOMPILING_EMULATOR)
    unset(UNI20_MPFR_HAS_TLS CACHE)
    check_cxx_source_runs([=[
      #include <mpfr.h>
      int main() { return mpfr_buildopt_tls_p() ? 0 : 1; }
    ]=] UNI20_MPFR_HAS_TLS)
    if(NOT UNI20_MPFR_HAS_TLS)
      message(FATAL_ERROR "Uni20 requires a thread-safe MPFR build (--enable-thread-safe).")
    endif()
  else()
    message(STATUS "Cross-compiling: MPFR thread-local support will be checked at runtime")
  endif()

  add_library(uni20_mpfr INTERFACE)
  target_include_directories(uni20_mpfr SYSTEM INTERFACE "${UNI20_MPFR_INCLUDE_DIR}" "${UNI20_GMP_INCLUDE_DIR}")
  target_link_libraries(uni20_mpfr INTERFACE "${UNI20_MPFR_LIBRARY}" "${UNI20_GMP_LIBRARY}")
  mark_as_advanced(UNI20_GMP_INCLUDE_DIR UNI20_GMP_LIBRARY UNI20_MPFR_INCLUDE_DIR UNI20_MPFR_LIBRARY)
endfunction()
