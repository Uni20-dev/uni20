add_library(uni20_cli STATIC cli.cpp)
target_link_libraries(uni20_cli PUBLIC uni20_common uni20_core CLI11::CLI11)
