#!/usr/bin/env python3
"""Render the numerical matrix from an actual Google Test XML result."""

import argparse
from collections import Counter
from pathlib import Path
import xml.etree.ElementTree as ET


def report(source):
    cells = []
    for test in ET.parse(source).iter("testcase"):
        suite = test.attrib.get("classname", "")
        if not suite.startswith(("NumericalScalar/", "NumericalLinalg/", "NumericalKrylov/")):
            continue
        properties = {p.attrib["name"]: p.attrib["value"] for p in test.findall("properties/property")}
        status, reason = "passed", ""
        if test.find("failure") is not None:
            status, reason = "failed", test.find("failure").attrib.get("message", "")
        elif test.find("skipped") is not None:
            reason = test.find("skipped").attrib.get("message", "")
            # GTest prefixes skip messages with their source location.
            status = next((s for s in ("unsupported", "unavailable", "not_applicable") if f"{s}:" in reason), "unknown_skip")
            reason = reason.split(f"{status}:", 1)[-1].strip()
        elif test.attrib.get("status") != "run" or test.attrib.get("result") != "completed":
            status = "not_run"
        cells.append((suite.split("/")[0] + "." + test.attrib["name"],
                      properties.get("scalar", suite.split("/")[-1]),
                      properties.get("backend", "unknown"), status, reason))
    if not cells:
        raise ValueError("no numerical precision results found")

    counts = Counter(cell[3] for cell in cells)
    lines = ["# Numerical precision results", "",
             ", ".join(f"{n} {s}" for s, n in sorted(counts.items())) + ".", "",
             "This reports executed probes, not certification of an entire scalar type. "
             "Unsupported and unavailable cells are not passes. Only tests present in the input XML are reported.", "",
             "| Operation | Scalar / precision | Backend | Result | Reason |",
             "| --- | --- | --- | --- | --- |"]
    for cell in sorted(cells):
        lines.append("| " + " | ".join(str(value).replace("|", "\\|").replace("\n", " ") for value in cell) + " |")
    return "\n".join(lines) + "\n", bool(counts["failed"] or counts["unknown_skip"] or counts["not_run"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("xml", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    text, failed = report(args.xml)
    if args.output:
        args.output.write_text(text)
    else:
        print(text, end="")
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
