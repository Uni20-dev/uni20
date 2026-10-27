# Developer Documentation

This directory contains repository process, build metadata, test, review, and
documentation tooling guides.

- [Testing](testing.md)
- [Numerical precision validation](numerical_testing.md)
- [Using Uni20 from another CMake project](cmake_consumers.md)
- [Reviewing Changes](code_review.md)
- [Agent-Assisted Development](agent_assisted_development.md)
- [Doxygen](doxygen.md)
- [Build Information](build_information.md)
- [Performance Measurements](performance_measurements.md)
- [Using Git and VS Code (Traditional Chinese)](git_and_vscode_zh_tw.md)

Run `python3 scripts/check-docs.py` after adding, moving, or renaming
documentation. Docs PR CI runs the same check before generating Doxygen.
It checks the topic hierarchy, subsystem indexes, local links, code fences,
headings, repository-root `docs/...` references, and source and example module
README coverage/navigation. Internal source `detail/` trees are documented by
their owning module and do not require separate indexes; existing READMEs in
those trees still have their Markdown and links checked.

The checker reads working-tree contents of files listed by `git ls-files`.
Stage new documents before checking them. Untracked local notes are ignored,
and links to untracked files cannot satisfy a tracked document's references.
Run the checker's regression tests with
`python3 -m unittest discover -s tests/docs -v`.

Repository-wide coding and correctness rules remain in
[`AGENTS.md`](../../AGENTS.md). Machine-specific build trees and compiler paths
belong in untracked local instructions, not these portable guides.

Relevant implementations live in [scalar and build foundations](../../src/uni20/core/)
and [common diagnostics/documentation support](../../src/uni20/common/).
