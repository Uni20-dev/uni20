# tests/backend/CMakeLists.txt

if(UNI20_ENABLE_MPLAPACK_MPFR)
  add_test_module(mplapack_mpfr SOURCES test_mplapack_mpfr.cpp
    LIBS uni20_backend_mplapack_mpfr uni20_core)
endif()

if(TARGET uni20_backend_blas)
  add_subdirectory(blas)
endif()

if(TARGET uni20_backend_cuda)
  add_subdirectory(cuda)
endif()

if(TARGET uni20_backend_cublas)
  add_subdirectory(cublas)
endif()

if(TARGET uni20_backend_cusolver)
  add_subdirectory(cusolver)
endif()
