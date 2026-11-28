set(random_test_sources test_philox.cpp test_transforms.cpp test_stream.cpp test_fill.cpp)
if(TARGET uni20_backend_cuda)
  add_test_module(philox
    SOURCES ${random_test_sources} test_philox_cuda.cu test_transforms_cuda.cu test_stream_cuda.cu test_fill_cuda.cpp
    LIBS uni20_linalg uni20_backend_cuda
  )
  set_target_properties(uni20_philox_tests PROPERTIES CUDA_STANDARD 20 CUDA_STANDARD_REQUIRED ON)
else()
  add_test_module(philox SOURCES ${random_test_sources} LIBS uni20_linalg)
endif()
