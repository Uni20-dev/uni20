# test_cuda_peer_kernel.cu is listed in SOURCES so the combined uni20_tests also links it.
add_test_module(cuda_runtime
  SOURCES test_cuda_buffer.cpp test_cuda_runtime.cpp test_cuda_task_awaiters.cpp test_cuda_tensor.cpp
          test_debug_cuda_scheduler.cpp test_cuda_storage_types.cpp test_cuda_peer_kernel.cu
  LIBS uni20_common uni20_async uni20_backend_cuda
)
set_target_properties(uni20_cuda_runtime_tests PROPERTIES
  CUDA_STANDARD 20
  CUDA_STANDARD_REQUIRED ON
)

add_test_module(cuda_tbb_scheduler
  SOURCES test_tbb_cuda_scheduler.cpp
  LIBS uni20_common uni20_async uni20_backend_cuda TBB::tbb
)
