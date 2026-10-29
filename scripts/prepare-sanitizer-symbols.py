#!/usr/bin/env python3
"""Prepare local ELF debug symbols before running sanitizers, never during tests.

The library and debug file must have the same GNU build ID. Supply an existing
debug file or an explicit debuginfod server. A verified cached file is reused.
Print the generated llvm-symbolizer wrapper path for ASAN_SYMBOLIZER_PATH.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import tempfile
from urllib.request import urlopen


def build_id(path):
    result = subprocess.run(['readelf', '--notes', str(path)], check=True,
                            capture_output=True, text=True, timeout=20)
    matches = set(re.findall(r'Build ID:\s*([0-9a-fA-F]+)', result.stdout))
    if len(matches) != 1:
        raise ValueError(f'{path}: expected exactly one GNU build ID')
    return matches.pop().lower()


def has_debug_info(path):
    result = subprocess.run(['readelf', '--sections', '--wide', str(path)], check=True,
                            capture_output=True, text=True, timeout=20)
    return re.search(r'\.(?:zdebug_info|debug_info)\b', result.stdout) is not None


def file_sha256(path):
    hasher = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(65536), b''):
            hasher.update(chunk)
    return hasher.hexdigest()


def prepare(library, output, symbolizer, debug_file=None, server=None):
    library = Path(library).resolve(strict=True)
    symbolizer_path = shutil.which(str(symbolizer))
    if symbolizer_path is None:
        raise ValueError(f'cannot find symbolizer: {symbolizer}')
    symbolizer_path = str(Path(symbolizer_path).absolute())
    identity = build_id(library)
    output = Path(output).resolve()
    debug_root = output / 'debug'
    target = debug_root / '.build-id' / identity[:2] / (identity[2:] + '.debug')
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if build_id(target) != identity:
            raise ValueError(f'cached debug file has a different build ID: {target}')
        if not has_debug_info(target):
            raise ValueError(f'cached file has no DWARF debug information: {target}')
    else:
        if debug_file is None and server is None:
            raise ValueError('no cached symbols: provide --debug-file or --server')
        if server is not None and not server.startswith('https://'):
            raise ValueError('the debuginfod server must use HTTPS')
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(dir=target.parent, delete=False) as stream:
                temporary = Path(stream.name)
                if debug_file is not None:
                    with Path(debug_file).open('rb') as source:
                        shutil.copyfileobj(source, stream)
                else:
                    url = server.rstrip('/') + '/buildid/' + identity + '/debuginfo'
                    with urlopen(url, timeout=30) as source:
                        shutil.copyfileobj(source, stream)
            if build_id(temporary) != identity:
                raise ValueError('debug-file build ID does not match the loaded library')
            if not has_debug_info(temporary):
                raise ValueError('file has no DWARF debug information')
            os.replace(temporary, target)
            temporary = None
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
    # An explicit wrapper makes the search path independent of sanitizer
    # runtime handling of LLVM_SYMBOLIZER_OPTS. No network access is allowed
    # when the sanitizer invokes this wrapper.
    wrapper = output / 'llvm-symbolizer'
    command = [symbolizer_path, '--debug-file-directory=' + str(debug_root)]
    wrapper.write_text('#!/bin/sh\nexport DEBUGINFOD_URLS=\nexec ' + shlex.join(command) + ' "$@"\n')
    wrapper.chmod(0o755)
    receipt = {'library': str(library), 'build_id': identity, 'debug_file': str(target),
               'debug_sha256': file_sha256(target),
               'symbolizer': symbolizer_path, 'wrapper': str(wrapper)}
    (output / 'symbols.json').write_text(json.dumps(receipt, indent=2) + '\n')
    return wrapper


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--library', required=True, type=Path)
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument('--symbolizer', default='llvm-symbolizer')
    source = parser.add_mutually_exclusive_group()
    source.add_argument('--debug-file', type=Path)
    source.add_argument('--server', help='explicit HTTPS debuginfod server, used during preparation only')
    args = parser.parse_args()
    try:
        wrapper = prepare(args.library, args.output_dir, args.symbolizer, args.debug_file, args.server)
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        parser.exit(1, f'symbol preparation failed: {error}\n')
    print(wrapper)


if __name__ == '__main__':
    main()
