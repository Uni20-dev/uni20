add_test_module(numerical_precision
  SOURCES test_scalar_precision.cpp test_linalg_precision.cpp test_lu_precision.cpp test_krylov_precision.cpp test_precision_registry.cpp
  LIBS uni20_linalg uni20_krylov)

find_package(Python3 QUIET COMPONENTS Interpreter)
if(Python3_Interpreter_FOUND)
  add_test(NAME NumericalCoverage.ReportCompleteness
    COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/test_precision_report.py")
endif()
