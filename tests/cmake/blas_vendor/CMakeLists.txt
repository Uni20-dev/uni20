cmake_minimum_required(VERSION 3.28)
# Exercise DLL/import-library metadata on every host.
# No compiler or Windows runtime is needed by this configuration-only fixture.
set(CMAKE_SYSTEM_NAME Windows)
project(blas_vendor_contract LANGUAGES NONE)
if(vendor_policy)
  cmake_policy(SET CMP0200 ${vendor_policy})
endif()

foreach(vendor_case IN ITEMS plain exact mapped mapped_empty mapped_list_empty
    default_location fallback no_config implib implib_mapped implib_only
    implib_default implib_default_mapped
    implib_runtime_default implib_runtime_default_mapped opaque)
  add_subdirectory(case "${vendor_case}")
endforeach()
