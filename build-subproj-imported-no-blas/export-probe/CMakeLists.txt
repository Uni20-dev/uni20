add_library(consumer_export_probe INTERFACE)
target_include_directories(consumer_export_probe INTERFACE
  "$<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include-one;${CMAKE_CURRENT_BINARY_DIR}/include-two>"
  "$<INSTALL_INTERFACE:include>")
relax_warnings(consumer_export_probe)
install(TARGETS consumer_export_probe EXPORT consumer_export_probe)
install(EXPORT consumer_export_probe DESTINATION lib/cmake/consumer_export_probe)
