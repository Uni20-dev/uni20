"""Exercise documentation checks in small, independently tracked repositories."""

from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest


CHECKER = Path(__file__).resolve().parents[2] / "scripts" / "check-docs.py"


class DocumentationChecks(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="uni20-docs-test-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.guide = Path("docs") / "guide"
        self.page = self.guide / "current.md"
        self.source = Path("src") / "uni20"
        self.git("init", "--quiet")
        (self.root / "scripts").mkdir()
        shutil.copyfile(CHECKER, self.root / "scripts" / "check-docs.py")
        self.write(Path("docs") / "README.md", "# Docs\n\n[Guide](guide/)\n")
        self.write(self.guide / "README.md", "# Guide\n\n[Current](current.md)\n")
        self.write(self.page, "# Current\n")
        self.write(self.source / "README.md", "# Source\n\n[Guide](../../docs/guide/)\n")
        self.write(Path("examples") / "README.md", "# Examples\n\n[Guide](../docs/guide/)\n")
        self.git("add", "--all")

    def git(self, *arguments):
        return subprocess.run(
            ["git", *arguments], cwd=self.root, check=True, capture_output=True, text=True
        )

    def write(self, path, text):
        target = self.root / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text)

    def check(self):
        return subprocess.run(
            [sys.executable, "scripts/check-docs.py"], cwd=self.root,
            capture_output=True, text=True,
        )

    def assert_valid(self):
        result = self.check()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def assert_invalid(self, diagnostic):
        result = self.check()
        self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
        self.assertIn(diagnostic, result.stdout)

    def test_valid_tracked_documentation(self):
        self.assert_valid()

    def test_repository_relative_paths_are_normalized(self):
        target = (self.guide / ".." / "guide" / "current.md").as_posix()
        self.write(self.page, f"# Current\n\n[Guide](/{target})\n\nSee `{target}`.\n")
        self.assert_valid()

    def test_internal_detail_directory_needs_no_separate_index(self):
        header = self.source / "detail" / "helper.hpp"
        self.write(header, "#pragma once\n")
        self.git("add", "--", str(header))
        self.assert_valid()

    def test_untracked_notes_and_source_directories_are_ignored(self):
        self.write(Path("docs") / "scratch.md", "# Scratch\n\n[Missing](missing.md)\n")
        self.write(self.guide / "notes" / "README.md", "#Bad heading\n")
        self.write(self.source / "local_experiment" / "helper.hpp", "#pragma once\n")
        self.assert_valid()

    def test_working_tree_broken_link_is_rejected(self):
        self.write(self.page, "# Current\n\n[Missing](missing.md)\n")
        self.assert_invalid("missing or untracked link target missing.md")

    def test_untracked_target_cannot_satisfy_a_tracked_link(self):
        self.write(self.page, "# Current\n\n[Local](local.txt)\n")
        self.write(self.guide / "local.txt", "Local notes\n")
        self.assert_invalid("missing or untracked link target local.txt")

    def test_newly_staged_document_requires_an_index_entry(self):
        new_page = self.guide / "new.md"
        self.write(new_page, "# New\n")
        self.git("add", "--", str(new_page))
        self.assert_invalid("new.md: not named in")

    def test_source_module_requires_an_index(self):
        header = self.source / "module" / "value.hpp"
        self.write(header, "#pragma once\n")
        self.git("add", "--", str(header))
        self.assert_invalid("module: missing README.md")

    def test_existing_detail_readme_links_are_checked(self):
        readme = self.source / "detail" / "README.md"
        self.write(readme, "# Details\n\n[Missing](missing.md)\n")
        self.git("add", "--", str(readme))
        self.assert_invalid("missing or untracked link target missing.md")

    def test_root_reference_cannot_resolve_to_untracked_note(self):
        note = self.guide / "local.md"
        self.write(note, "# Local\n")
        self.write(self.page, "# Current\n\nSee " + note.as_posix() + ".\n")
        self.assert_invalid("missing repository-root reference " + note.as_posix())


if __name__ == "__main__":
    unittest.main()
