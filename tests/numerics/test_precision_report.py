"""An incomplete numerical run must never look like complete passing coverage."""

import importlib.util
from pathlib import Path
import tempfile
import unittest
import xml.etree.ElementTree as ET

spec = importlib.util.spec_from_file_location(
    "precision_report", Path(__file__).resolve().parents[2] / "scripts/report-numerical-precision.py")
reporter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reporter)


class PrecisionReportTest(unittest.TestCase):
    def setUp(self):
        self.root = ET.Element("testsuites")
        self.registry = self.add_test("NumericalCoverage", "RegisteredProbesMatchDeclaredMatrix")
        properties = ET.SubElement(self.registry, "properties")
        ET.SubElement(properties, "property", name="precision_matrix_v1", value=(
            "NumericalScalar.Probe\tRealFloat64\tscalar\tready\t\n"
            "NumericalScalar.Probe\tRealMp128\tscalar\tunsupported\tnot implemented\n"))
        self.probe = self.add_test("NumericalScalar/RealFloat64", "Probe")

    def add_test(self, suite, name):
        return ET.SubElement(self.root, "testcase", classname=suite, name=name, status="run", result="completed")

    def render(self):
        with tempfile.TemporaryDirectory() as directory:
            xml = Path(directory) / "results.xml"
            ET.ElementTree(self.root).write(xml)
            return reporter.report(xml)

    def test_exclusions_are_reported_separately(self):
        text, failed = self.render()
        self.assertFalse(failed)
        self.assertIn("1 passed, 1 unsupported", text)
        self.assertIn("not implemented", text)

    def test_missing_expected_result_fails(self):
        self.root.remove(self.probe)
        text, failed = self.render()
        self.assertTrue(failed)
        self.assertIn("expected probe missing from XML", text)

    def test_unexecuted_result_fails(self):
        self.probe.set("status", "notrun")
        text, failed = self.render()
        self.assertTrue(failed)
        self.assertIn("registered probe was not executed", text)

    def test_skipping_expected_probe_fails(self):
        ET.SubElement(self.probe, "skipped", message="unsupported: accidentally lost support")
        text, failed = self.render()
        self.assertTrue(failed)
        self.assertIn("unexpected_skip", text)

    def test_failure_is_preserved(self):
        ET.SubElement(self.probe, "failure", message="lost precision")
        text, failed = self.render()
        self.assertTrue(failed)
        self.assertIn("lost precision", text)

    def test_excluded_probe_cannot_count_as_a_pass(self):
        self.add_test("NumericalScalar/RealMp128", "Probe")
        text, failed = self.render()
        self.assertTrue(failed)
        self.assertIn("probe registered despite declared unsupported coverage", text)

    def test_undeclared_probe_fails(self):
        self.add_test("NumericalScalar/RealFloat64", "UnknownProbe")
        text, failed = self.render()
        self.assertTrue(failed)
        self.assertIn("probe absent from coverage declaration", text)

    def test_missing_manifest_is_an_error(self):
        self.root.remove(self.registry)
        with self.assertRaisesRegex(ValueError, "missing NumericalCoverage"):
            self.render()

    def test_registry_failure_is_preserved(self):
        ET.SubElement(self.registry, "failure", message="expected registration missing")
        text, failed = self.render()
        self.assertTrue(failed)
        self.assertIn("expected registration missing", text)


if __name__ == "__main__":
    unittest.main()
