if(NOT TARGET LAPACK::LAPACK)
  find_package(LAPACK REQUIRED)
endif()

add_library(uni20_backend_lapack INTERFACE)

if(TARGET LAPACK::LAPACK)
  target_link_libraries(uni20_backend_lapack INTERFACE LAPACK::LAPACK)
else()
  target_link_libraries(uni20_backend_lapack INTERFACE ${LAPACK_LIBRARIES})
  if(LAPACK_LINKER_FLAGS)
    target_link_options(uni20_backend_lapack INTERFACE ${LAPACK_LINKER_FLAGS})
  endif()
endif()
