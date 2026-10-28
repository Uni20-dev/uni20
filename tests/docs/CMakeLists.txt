find_package(Python3 REQUIRED COMPONENTS Interpreter)

set(snippet_dir "${CMAKE_CURRENT_BINARY_DIR}/generated")
set(snippets
  "${snippet_dir}/readme_product_step.inc"
  "${snippet_dir}/readme_product_step_calls.inc")
set(extractor "${PROJECT_SOURCE_DIR}/scripts/extract-readme-snippets.py")
add_custom_command(
  OUTPUT ${snippets}
  COMMAND Python3::Interpreter "${extractor}" "${PROJECT_SOURCE_DIR}/README.md" "${snippet_dir}"
  DEPENDS "${extractor}" "${PROJECT_SOURCE_DIR}/README.md"
  COMMENT "Extracting tested C++ snippets from README.md"
  VERBATIM)
add_custom_target(uni20_readme_snippets_generate DEPENDS ${snippets})

# Carry the generated include path and build dependency into both the module
# executable and the optional combined test executable.
add_library(uni20_readme_snippets INTERFACE)
target_include_directories(uni20_readme_snippets INTERFACE "${snippet_dir}")
add_dependencies(uni20_readme_snippets uni20_readme_snippets_generate)
add_test_module(docs SOURCES test_readme.cpp
  LIBS uni20_linalg uni20_async uni20_readme_snippets)

add_test(NAME Documentation.SnippetExtraction
  COMMAND Python3::Interpreter -m unittest discover
    -s "${CMAKE_CURRENT_SOURCE_DIR}" -p test_extract_readme_snippets.py -v)
