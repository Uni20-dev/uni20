function(check name expected)
  execute_process(COMMAND "${CMAKE_COMMAND}" "-Uexpect_*"
    -S "${UNI20_SOURCE_DIR}/tests/cmake/scalars" -B "${test_binary_dir}/${name}"
    "-DUNI20_SOURCE_DIR=${UNI20_SOURCE_DIR}" "-DCMAKE_CXX_COMPILER=${compiler}"
    ${ARGN}
    RESULT_VARIABLE result OUTPUT_VARIABLE output ERROR_VARIABLE error)
  string(REGEX REPLACE "[ \t\r\n]+" " " diagnostic "${output}${error}")
  if(expected STREQUAL "success")
    if(NOT result EQUAL 0)
      message(FATAL_ERROR "${name}:\n${output}\n${error}")
    endif()
  elseif(result EQUAL 0 OR NOT diagnostic MATCHES "${expected}")
    message(FATAL_ERROR "${name}: expected failure '${expected}', got:\n${output}\n${error}")
  endif()
endfunction()

check(defaults success -Dexpect_FLOAT128=OFF -Dexpect_MPREAL=OFF)
check(disabled success -DUNI20_ENABLE_FLOAT80=OFF -DUNI20_ENABLE_FLOAT128=OFF
  -DUNI20_ENABLE_MPREAL=OFF -Dexpect_FLOAT80=OFF -Dexpect_FLOAT128=OFF -Dexpect_MPREAL=OFF)
check(invalid "expected AUTO, ON or OFF" -DUNI20_ENABLE_FLOAT128=perhaps)
check(obsolete "has been removed" -DUNI20_ENABLE_MPFR=ON)
check(auto_missing success -Dcase=missing -DUNI20_ENABLE_MPREAL=AUTO
  -DUNI20_ENABLE_FLOAT128=AUTO -Dexpect_MPREAL=OFF -Dexpect_FLOAT128=OFF)
check(auto_no_fetch success -DUNI20_ENABLE_FLOAT128=auto -DUNI20_USE_SYSTEM_MPLAPACK=off
  -Dexpect_FLOAT128=OFF)
check(required_missing "requires installed GMP and MPFR" -Dcase=missing -DUNI20_ENABLE_MPREAL=ON)
check(empty_backend "requires at least one available optional scalar"
  -DUNI20_ENABLE_FLOAT80=OFF -DUNI20_BACKEND_MPLAPACK=ON)

# A repeated configure must not retain a previous availability result.
check(reconfigure success -DUNI20_ENABLE_FLOAT80=AUTO)
check(reconfigure success -DUNI20_ENABLE_FLOAT80=OFF -Dexpect_FLOAT80=OFF)
if(native_float80)
  check(no_float80_auto success -Dcase=no_float80 -DUNI20_ENABLE_FLOAT80=AUTO -Dexpect_FLOAT80=OFF)
  check(no_float80_required "requires native x87" -Dcase=no_float80 -DUNI20_ENABLE_FLOAT80=ON)
endif()
if(has_float128 AND mplapack_source STREQUAL "system")
  check(float128_auto success -DUNI20_ENABLE_FLOAT128=AUTO -Dexpect_FLOAT128=ON
    -DUNI20_USE_SYSTEM_MPLAPACK=ON "-Dmplapack_DIR=${mplapack_dir}")
  check(float128_off success -DUNI20_ENABLE_FLOAT128=OFF -Dexpect_FLOAT128=OFF
    "-Dmplapack_DIR=${mplapack_dir}")
endif()
if(has_mpreal)
  set(paths "-DUNI20_GMP_INCLUDE_DIR=${gmp_include}" "-DUNI20_GMP_LIBRARY=${gmp_library}"
    "-DUNI20_MPFR_INCLUDE_DIR=${mpfr_include}" "-DUNI20_MPFR_LIBRARY=${mpfr_library}"
    "-DUNI20_MPC_INCLUDE_DIR=${mpc_include}" "-DUNI20_MPC_LIBRARY=${mpc_library}")
  check(mpreal_auto success -DUNI20_ENABLE_MPREAL=AUTO -Dexpect_MPREAL=ON ${paths})
  check(mpreal_on success -DUNI20_ENABLE_MPREAL=ON -Dexpect_MPREAL=ON ${paths})
  check(broken_mpc_auto success -Dcase=broken_mpc -DUNI20_ENABLE_MPREAL=AUTO -Dexpect_MPREAL=OFF ${paths})
  check(broken_mpc_on "Cannot compile and link MPC" -Dcase=broken_mpc -DUNI20_ENABLE_MPREAL=ON ${paths})
endif()
