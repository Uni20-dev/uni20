# CLI11 is opt-in and must never become a numerical target's dependency.
if(TARGET CLI11::CLI11)
  set(UNI20_CLI11_SOURCE "parent" CACHE INTERNAL "CLI11 dependency source" FORCE)
  set(UNI20_DETECTED_CLI11 "parent" CACHE INTERNAL "CLI11 dependency source" FORCE)
else()
  uni20_add_dependency(
    NAME CLI11
    VERSION 2.7.2
    TARGET CLI11::CLI11
    REPO https://github.com/CLIUtils/CLI11.git
    TAG cbd58a3696887b34c70949aef21a71735a0c2ad5 # v2.7.2
    DEFAULTS
      "CLI11_BUILD_TESTS=OFF"
      "CLI11_BUILD_EXAMPLES=OFF"
      "CLI11_BUILD_DOCS=OFF"
      "CLI11_INSTALL=OFF"
  )
endif()
