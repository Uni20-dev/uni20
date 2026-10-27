#!/usr/bin/env python3
"""Extract the two marked README C++ blocks used by the documentation test."""

import argparse
from pathlib import Path


SNIPPETS = {
    "product-step": "readme_product_step.inc",
    "product-step-calls": "readme_product_step_calls.inc",
}


def extract_cpp_block(document, name):
    """Return a marked block verbatim and its first one-based source line."""
    lines = document.splitlines(keepends=True)
    marker = f"<!-- tested-snippet: {name} -->"
    matches = [i for i, line in enumerate(lines) if line.strip() == marker]
    if len(matches) != 1:
        raise ValueError(f"expected exactly one {marker}, found {len(matches)}")
    opening = matches[0] + 1
    while opening < len(lines) and not lines[opening].strip():
        opening += 1
    if opening == len(lines) or lines[opening].strip() != "```cpp":
        raise ValueError(f"{name}: marker must be followed by a ```cpp fence")
    start = opening + 1
    for closing in range(start, len(lines)):
        fence = lines[closing].strip()
        if fence == "```":
            code = "".join(lines[start:closing])
            if not code.strip():
                raise ValueError(f"{name}: empty C++ block")
            return code, start + 1
        if fence.startswith("```"):
            raise ValueError(f"{name}: unexpected fence before closing ```")
    raise ValueError(f"{name}: missing closing ``` fence")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("readme", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    document = args.readme.read_text(encoding="utf-8")
    # Validate every block before producing any outputs.
    try:
        blocks = {name: extract_cpp_block(document, name) for name in SNIPPETS}
    except ValueError as error:
        parser.error(f"{args.readme}: {error}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    source = str(args.readme.resolve()).replace("\\", "\\\\").replace('"', '\\"')
    for name, (code, line) in blocks.items():
        (args.output_dir / SNIPPETS[name]).write_text(
            f'// Generated from README.md; do not edit.\n#line {line} "{source}"\n{code}',
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
