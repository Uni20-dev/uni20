add_test_module(numerical_precision
  NO_PRETTY_TYPES # Preserve names such as RealMp128 instead of printing the C++ case type.
  SOURCES test_scalar_precision.cpp test_linalg_precision.cpp test_krylov_precision.cpp
  LIBS uni20_linalg uni20_krylov)
