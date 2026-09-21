if(UNI20_BACKEND_CUDA AND (NOT UNI20_ENABLE_CUDA))
  message(FATAL_ERROR "CUDA backend requested but CUDA support is disabled (UNI20_ENABLE_CUDA is OFF)")
endif()

find_package(CUDAToolkit REQUIRED)

if(NOT TARGET CUDA::cudart)
  message(FATAL_ERROR "CUDA backend requested but the CUDA runtime library is not available.")
endif()

add_library(uni20_backend_cuda STATIC
    cuda_error.cpp
    device.cpp
    initialization.cu
    runtime.cpp
)
set_target_properties(uni20_backend_cuda PROPERTIES
    CUDA_STANDARD 20
    CUDA_STANDARD_REQUIRED ON
)
target_link_libraries(uni20_backend_cuda PUBLIC uni20_common CUDA::cudart)
