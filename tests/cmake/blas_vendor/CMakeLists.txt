cmake_minimum_required(VERSION 3.24)
project(blas_vendor_contract LANGUAGES NONE)

foreach(vendor_case IN ITEMS plain exact mapped mapped_empty mapped_list_empty
    default_location fallback no_config implib implib_mapped implib_only opaque)
  add_subdirectory(case "${vendor_case}")
endforeach()
