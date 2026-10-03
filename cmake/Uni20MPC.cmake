include_guard(GLOBAL)

function(uni20_find_mpc)
  find_path(UNI20_MPC_INCLUDE_DIR NAMES mpc.h)
  find_library(UNI20_MPC_LIBRARY NAMES mpc)
  if(NOT UNI20_MPC_INCLUDE_DIR OR NOT UNI20_MPC_LIBRARY)
    message(FATAL_ERROR
      "UNI20_ENABLE_MPC requires installed MPC development files. On Ubuntu: sudo apt install libmpc-dev. "
      "Set CMAKE_PREFIX_PATH or UNI20_MPC_{INCLUDE_DIR,LIBRARY} for a custom installation.")
  endif()
  include(CheckCXXSourceCompiles)
  set(CMAKE_REQUIRED_INCLUDES "${UNI20_MPC_INCLUDE_DIR};${UNI20_MPFR_INCLUDE_DIR};${UNI20_GMP_INCLUDE_DIR}")
  set(CMAKE_REQUIRED_LIBRARIES "${UNI20_MPC_LIBRARY};${UNI20_MPFR_LIBRARY};${UNI20_GMP_LIBRARY}")
  set(CMAKE_TRY_COMPILE_TARGET_TYPE EXECUTABLE)
  unset(UNI20_MPC_LINKS CACHE)
  check_cxx_source_compiles([=[
    #include <mpc.h>
    int main() {
      mpc_t z; mpc_init2(z, 256); mpc_set_ui(z, 2, MPC_RNDNN);
      mpc_sqrt(z, z, MPC_RNDNN); mpc_clear(z);
    }
  ]=] UNI20_MPC_LINKS)
  if(NOT UNI20_MPC_LINKS)
    message(FATAL_ERROR "Cannot compile and link MPC with MPFR/GMP; check the selected headers and libraries.")
  endif()
  add_library(uni20_mpc INTERFACE)
  target_include_directories(uni20_mpc SYSTEM INTERFACE "${UNI20_MPC_INCLUDE_DIR}")
  target_link_libraries(uni20_mpc INTERFACE "${UNI20_MPC_LIBRARY}" uni20_mpfr)
  mark_as_advanced(UNI20_MPC_INCLUDE_DIR UNI20_MPC_LIBRARY)
endfunction()
