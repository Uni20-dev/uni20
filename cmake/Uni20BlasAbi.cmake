# CMake's imported numerical targets have no standard integer-ABI property.
# The parent must declare the ABI of any targets it supplies before Uni20
# overrides local discovery hints. This is a caller contract, not an ABI probe.
function(uni20_validate_parent_blas_abi required_bytes)
  foreach(provider IN ITEMS BLAS LAPACK)
    if(TARGET ${provider}::${provider} AND
        NOT "${BLA_SIZEOF_INTEGER}" STREQUAL "${required_bytes}")
      message(FATAL_ERROR
        "Existing ${provider}::${provider} requires an explicit "
        "BLA_SIZEOF_INTEGER=${required_bytes} matching UNI20_ILP64 before adding Uni20. "
        "The parent must declare the actual integer ABI of its BLAS/LAPACK targets; "
        "missing, ANY, or conflicting declarations are not supported "
        "(received '${BLA_SIZEOF_INTEGER}').")
    endif()
  endforeach()
endfunction()
