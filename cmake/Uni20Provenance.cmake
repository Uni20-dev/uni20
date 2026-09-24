# Build-time source identity for Uni20 and add_subdirectory consumers. No runtime Git access.
function(uni20_target_provenance target)
  cmake_parse_arguments(P "" "HEADER;NAMESPACE;SOURCE_DIR;REVISION" "" ${ARGN})
  if(NOT TARGET "${target}" OR NOT P_HEADER OR NOT P_NAMESPACE OR NOT P_SOURCE_DIR)
    message(FATAL_ERROR "uni20_target_provenance requires a target, HEADER, NAMESPACE and SOURCE_DIR")
  endif()
  if(NOT P_NAMESPACE MATCHES "^[A-Za-z_][A-Za-z_0-9]*(::[A-Za-z_][A-Za-z_0-9]*)*$")
    message(FATAL_ERROR "Invalid provenance namespace: ${P_NAMESPACE}")
  endif()
  get_filename_component(_source "${P_SOURCE_DIR}" ABSOLUTE)
  set(_include "${CMAKE_CURRENT_BINARY_DIR}/${target}_provenance")
  set(_header "${_include}/${P_HEADER}")
  add_custom_target(${target}_source_provenance
    COMMAND "${CMAKE_COMMAND}" "-DSOURCE_DIR=${_source}" "-DOUTPUT=${_header}"
      "-DNAMESPACE=${P_NAMESPACE}" "-DREVISION_OVERRIDE=${P_REVISION}"
      -P "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/WriteSourceProvenance.cmake"
    BYPRODUCTS "${_header}"
    VERBATIM)
  add_dependencies(${target} ${target}_source_provenance)
  get_target_property(_type ${target} TYPE)
  if(_type STREQUAL "INTERFACE_LIBRARY")
    target_include_directories(${target} INTERFACE "${_include}")
  else()
    target_include_directories(${target} PRIVATE "${_include}")
  endif()
endfunction()
