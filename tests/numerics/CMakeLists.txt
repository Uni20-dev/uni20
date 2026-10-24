add_test_module(numerical_precision
  SOURCES test_refinement_precision.cpp test_rank_aware_precision.cpp test_least_squares_precision.cpp test_decomposition_precision.cpp test_scalar_precision.cpp test_special_precision.cpp test_pi_math_precision.cpp test_exponential_math_precision.cpp test_generic_precision.cpp test_linalg_precision.cpp test_lu_precision.cpp test_krylov_precision.cpp test_gmres_precision.cpp test_precision_registry.cpp
  LIBS uni20_linalg uni20_krylov)

find_package(Python3 QUIET COMPONENTS Interpreter)
if(Python3_Interpreter_FOUND)
  add_test(NAME NumericalCoverage.ReportCompleteness
    COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/test_precision_report.py")
endif()
