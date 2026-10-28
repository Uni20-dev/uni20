"""Check that snippet extraction cannot silently omit or substitute the code."""

import importlib.util
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "extract-readme-snippets.py"
SPEC = importlib.util.spec_from_file_location("extract_readme_snippets", SCRIPT)
EXTRACTOR = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(EXTRACTOR)


class ReadmeSnippetExtraction(unittest.TestCase):
    block = "<!-- tested-snippet: product-step -->\n\n```cpp\n  example(); // verbatim\n```\n"

    def test_preserves_code_and_source_line(self):
        document = "# Example\n\n" + self.block + "\n```cpp\nunrelated();\n```\n"
        self.assertEqual(
            EXTRACTOR.extract_cpp_block(document, "product-step"),
            ("  example(); // verbatim\n", 6),
        )

    def test_missing_or_duplicate_marker_fails(self):
        for document in ("```cpp\nexample();\n```", self.block + self.block):
            with self.subTest(document=document):
                with self.assertRaisesRegex(ValueError, "exactly one"):
                    EXTRACTOR.extract_cpp_block(document, "product-step")

    def test_missing_or_wrong_opening_fence_fails(self):
        for suffix in ("", "\n", "```python\npass\n```\n", "prose\n```cpp\ncode\n```\n"):
            with self.subTest(suffix=suffix):
                with self.assertRaisesRegex(ValueError, "followed by a ```cpp"):
                    EXTRACTOR.extract_cpp_block("<!-- tested-snippet: product-step -->\n" + suffix, "product-step")

    def test_unclosed_block_fails(self):
        with self.assertRaisesRegex(ValueError, "missing closing"):
            EXTRACTOR.extract_cpp_block(self.block.removesuffix("```\n"), "product-step")
        with self.assertRaisesRegex(ValueError, "unexpected fence"):
            EXTRACTOR.extract_cpp_block(self.block.replace("```\n", "```cpp\n"), "product-step")

    def test_empty_block_fails(self):
        with self.assertRaisesRegex(ValueError, "empty C\\+\\+ block"):
            EXTRACTOR.extract_cpp_block(self.block.replace("  example(); // verbatim", "  "), "product-step")

    def test_command_generates_both_blocks_and_rejects_missing_calls(self):
        with tempfile.TemporaryDirectory(prefix="uni20-readme-test-") as directory:
            root = Path(directory)
            readme = root / "README.md"
            output = root / "generated"
            readme.write_text(self.block + self.block.replace("product-step", "product-step-calls"), encoding="utf-8")
            command = [sys.executable, str(SCRIPT), str(readme), str(output)]
            result = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            for filename in EXTRACTOR.SNIPPETS.values():
                generated = (output / filename).read_text(encoding="utf-8")
                self.assertIn("#line ", generated)
                self.assertTrue(generated.endswith("  example(); // verbatim\n"))
            readme.write_text(self.block, encoding="utf-8")
            result = subprocess.run(command, capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("product-step-calls", result.stderr)


if __name__ == "__main__":
    unittest.main()
