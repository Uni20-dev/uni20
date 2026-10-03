#!/usr/bin/env python3
"""Join declared numerical coverage to a complete Google Test XML result."""

import argparse
from collections import Counter
from pathlib import Path
import xml.etree.ElementTree as ET


def test_result(test):
    failure = test.find("failure")
    if failure is None:
        failure = test.find("error")
    if failure is not None:
        return "failed", failure.attrib.get("message", "")
    skipped = test.find("skipped")
    if skipped is not None:
        return "unexpected_skip", skipped.attrib.get("message", "")
    if test.attrib.get("status") != "run" or test.attrib.get("result") != "completed":
        return "not_run", "registered probe was not executed"
    return "passed", ""


def report(source):
    root = ET.parse(source)
    actual = {}
    registry = None
    for test in root.iter("testcase"):
        suite = test.attrib.get("classname", "")
        if suite == "NumericalCoverage" and test.attrib["name"] == "RegisteredProbesMatchDeclaredMatrix":
            registry = test
        if not suite.startswith(("NumericalScalar/", "NumericalLinalg/", "NumericalKrylov/")):
            continue
        properties = {p.attrib["name"]: p.attrib["value"] for p in test.findall("properties/property")}
        key = (suite.split("/")[0] + "." + test.attrib["name"],
               properties.get("scalar", suite.split("/")[-1]))
        if key in actual:
            raise ValueError(f"duplicate numerical result: {key}")
        actual[key] = (properties.get("backend", "unknown"), *test_result(test))

    if registry is None:
        raise ValueError("missing NumericalCoverage registry result; run the unfiltered numerical executable")
    registry_status, registry_reason = test_result(registry)
    manifest = registry.find("properties/property[@name='precision_matrix_v1']")
    if manifest is None or not manifest.attrib.get("value"):
        raise ValueError("missing precision matrix; the coverage registry test must execute")

    cells = []
    seen = set()
    for row in manifest.attrib["value"].splitlines():
        operation, scalar, backend, state, reason = row.split("\t", 4)
        key = (operation, scalar)
        if key in seen:
            raise ValueError(f"duplicate declared probe: {key}")
        seen.add(key)
        result = actual.pop(key, None)
        if state == "ready":
            if result is None:
                status, reason = "not_run", "expected probe missing from XML"
            else:
                backend, status, reason = result
        elif state in ("unsupported", "unavailable", "not_applicable"):
            status = state
            if result is not None:
                status, reason = "unexpected_test", f"probe registered despite declared {state} coverage"
        else:
            raise ValueError(f"unknown coverage state: {state}")
        cells.append((operation, scalar, backend, status, reason))

    for (operation, scalar), (backend, status, reason) in actual.items():
        cells.append((operation, scalar, backend, "unexpected_test", "probe absent from coverage declaration"))
    if registry_status != "passed":
        cells.append(("NumericalCoverage.RegisteredProbesMatchDeclaredMatrix", "all", "registry",
                      registry_status, registry_reason))

    counts = Counter(cell[3] for cell in cells)
    lines = ["# Numerical precision results", "",
             ", ".join(f"{n} {s}" for s, n in sorted(counts.items())) + ".", "",
             "Results join the complete declared coverage matrix to actual execution. "
             "Unsupported, unavailable and inapplicable combinations are not registered tests or passes. "
             "Missing expected results are errors.", "",
             "| Operation | Scalar / precision | Backend | Result | Reason |",
             "| --- | --- | --- | --- | --- |"]
    for cell in sorted(cells):
        lines.append("| " + " | ".join(str(value).replace("|", "\\|").replace("\n", " ") for value in cell) + " |")
    failed = any(counts[state] for state in ("failed", "unexpected_skip", "not_run", "unexpected_test"))
    return "\n".join(lines) + "\n", failed


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
