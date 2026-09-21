# Each failure must be the ABI diagnostic, not an unrelated configure error.
foreach(provider IN ITEMS BLAS LAPACK)
  foreach(required_bytes IN ITEMS 4 8)
    foreach(declaration IN ITEMS 4 8 missing ANY invalid)
      execute_process(COMMAND "${CMAKE_COMMAND}" --fresh
        -S "${CMAKE_CURRENT_LIST_DIR}/blas_abi"
        -B "${test_binary_dir}/${provider}-${required_bytes}-${declaration}"
        "-DUNI20_SOURCE_DIR=${UNI20_SOURCE_DIR}"
        "-Dprovider=${provider}" "-Drequired_bytes=${required_bytes}"
        "-Ddeclaration=${declaration}"
        RESULT_VARIABLE result OUTPUT_VARIABLE output ERROR_VARIABLE error)
      if(declaration STREQUAL required_bytes)
        if(NOT result EQUAL 0)
          message(FATAL_ERROR "Matching ABI was rejected: ${output}\n${error}")
        endif()
      elseif(result EQUAL 0 OR NOT error MATCHES "Existing ${provider}::${provider} requires an explicit")
        message(FATAL_ERROR "Expected ABI rejection for ${provider}/${required_bytes}/${declaration}: ${output}\n${error}")
      endif()
    endforeach()
  endforeach()
endforeach()
