"""Hermetic build-ID, cache and quoting checks for explicit symbol preparation."""
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

script = Path(__file__).resolve().parents[2] / 'scripts/prepare-sanitizer-symbols.py'
spec = importlib.util.spec_from_file_location('prepare_symbols', script)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class PrepareSymbols(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.library = self.root/'lib.so'
        self.library.write_bytes(b'library')
        self.debug = self.root/'source.debug'
        self.debug.write_bytes(b'debug')
        self.output = self.root/"symbols with ' quotes"
        debug_check = patch.object(module, 'has_debug_info', return_value=True)
        debug_check.start()
        self.addCleanup(debug_check.stop)

    def identities(self, path):
        return 'abcdef' if Path(path).read_bytes() in (b'library', b'debug') else '123456'

    def test_local_file_and_verified_cache(self):
        with patch.object(module, 'build_id', side_effect=self.identities):
            wrapper = module.prepare(self.library, self.output, '/bin/echo', self.debug)
            self.debug.unlink()
            with patch.object(module, 'urlopen', side_effect=AssertionError('unexpected network')):
                self.assertEqual(module.prepare(self.library, self.output, '/bin/echo'), wrapper)
        receipt = json.loads((self.output/'symbols.json').read_text())
        self.assertEqual(receipt['build_id'], 'abcdef')
        done = subprocess.run([str(wrapper), 'CODE path with spaces 0x10'], capture_output=True, text=True, check=True)
        self.assertIn(str(self.output/'debug'), done.stdout)
        self.assertIn('CODE path with spaces 0x10', done.stdout)

    def test_mismatched_file_never_enters_cache(self):
        self.debug.write_bytes(b'wrong')
        with patch.object(module, 'build_id', side_effect=self.identities):
            with self.assertRaisesRegex(ValueError, 'does not match'):
                module.prepare(self.library, self.output, '/bin/echo', self.debug)
        self.assertFalse((self.output/'llvm-symbolizer').exists())
        self.assertEqual(list((self.output/'debug/.build-id/ab').iterdir()), [])

    def test_corrupt_cache_is_an_error(self):
        with patch.object(module, 'build_id', side_effect=self.identities):
            module.prepare(self.library, self.output, '/bin/echo', self.debug)
            (self.output/'debug/.build-id/ab/cdef.debug').write_bytes(b'wrong')
            with self.assertRaisesRegex(ValueError, 'cached debug file'):
                module.prepare(self.library, self.output, '/bin/echo', self.debug)

    def test_explicit_server_only(self):
        with patch.object(module, 'build_id', side_effect=self.identities):
            with self.assertRaisesRegex(ValueError, 'provide --debug-file or --server'):
                module.prepare(self.library, self.output, '/bin/echo')
            with self.assertRaisesRegex(ValueError, 'HTTPS'):
                module.prepare(self.library, self.output, '/bin/echo', server='http://example.invalid')
            with patch.object(module, 'urlopen', return_value=io.BytesIO(b'debug')) as download:
                module.prepare(self.library, self.output, '/bin/echo', server='https://example.invalid/')
                download.assert_called_once_with('https://example.invalid/buildid/abcdef/debuginfo', timeout=30)

    def test_matching_stripped_library_is_not_debug_info(self):
        with patch.object(module, 'build_id', side_effect=self.identities), \
             patch.object(module, 'has_debug_info', return_value=False):
            with self.assertRaisesRegex(ValueError, 'no DWARF'):
                module.prepare(self.library, self.output, '/bin/echo', self.debug)
        self.assertFalse((self.output/'llvm-symbolizer').exists())

    def test_build_id_parser(self):
        with patch.object(module.subprocess, 'run', return_value=subprocess.CompletedProcess([],0,' Build ID: aBcD1234\n')):
            self.assertEqual(module.build_id('lib.so'), 'abcd1234')
        with patch.object(module.subprocess, 'run', return_value=subprocess.CompletedProcess([],0,'no identity')):
            with self.assertRaisesRegex(ValueError, 'GNU build ID'):
                module.build_id('lib.so')


if __name__ == '__main__':
    unittest.main()
