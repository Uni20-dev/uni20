# tests/tensor/CMakeLists.txt

add_test_module(tensor
  SOURCES
    test_tensor.cpp
    test_tensor_conjugate.cpp
    test_tensor_output.cpp
    test_tensor_initialization.cpp
  LIBS uni20_common uni20_core uni20_linalg
)

add_test_module(tensor_async
  SOURCES test_tensor_async.cpp
  LIBS uni20_linalg uni20_async
)
