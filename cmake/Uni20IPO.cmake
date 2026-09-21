# Directory-local defaults for targets created after this module is included.
# Never clear inherited IPO settings: the parent/toolchain owns those settings
# and the final link when Uni20 is embedded.
if(NOT UNI20_ENABLE_LTO)
  message(STATUS "Uni20 automatic LTO/IPO disabled (UNI20_ENABLE_LTO=OFF); inherited IPO settings unchanged")
  return()
endif()

get_property(_uni20_ipo_multi_config GLOBAL PROPERTY GENERATOR_IS_MULTI_CONFIG)
if(_uni20_ipo_multi_config)
  set(_uni20_ipo_available_configs ${CMAKE_CONFIGURATION_TYPES})
else()
  set(_uni20_ipo_available_configs "${CMAKE_BUILD_TYPE}")
endif()
list(TRANSFORM _uni20_ipo_available_configs TOUPPER)
list(FILTER _uni20_ipo_available_configs INCLUDE REGEX "^(RELEASE|RELWITHDEBINFO|MINSIZEREL|DEBUGOPT)$")
if(NOT _uni20_ipo_available_configs)
  message(STATUS "Uni20 automatic LTO/IPO skipped: no optimized configuration; inherited IPO settings unchanged")
  return()
endif()

# Check the languages used by Uni20 targets, rather than unrelated languages
# enabled by the parent or dependencies. CUDA IPO includes device LTO.
set(_uni20_ipo_languages CXX)
if(UNI20_ENABLE_CUDA)
  list(APPEND _uni20_ipo_languages CUDA)
endif()
include(CheckIPOSupported)
check_ipo_supported(RESULT _uni20_ipo_supported OUTPUT _uni20_ipo_error LANGUAGES ${_uni20_ipo_languages})
if(NOT _uni20_ipo_supported)
  message(STATUS "Uni20 automatic LTO/IPO unavailable for ${_uni20_ipo_languages}: ${_uni20_ipo_error}\n"
    "Inherited IPO settings unchanged")
  return()
endif()

# Per-configuration initializers work with both single- and multi-config
# generators. Do not enable Debug or modify the generic IPO initializer.
foreach(_uni20_ipo_config IN LISTS _uni20_ipo_available_configs)
  set(CMAKE_INTERPROCEDURAL_OPTIMIZATION_${_uni20_ipo_config} TRUE)
endforeach()
message(STATUS "Uni20 automatic LTO/IPO enabled for ${_uni20_ipo_available_configs} (${_uni20_ipo_languages})")
