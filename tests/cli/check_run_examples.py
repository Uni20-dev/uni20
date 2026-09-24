"""Check interoperable exports and staged configuration with independent stdlib readers."""
import csv
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from decimal import Decimal

examples = Path(sys.argv[1])


def run(name, *args, env=None, success=True):
    process = subprocess.run([str(examples / name), *map(str, args)], text=True, capture_output=True, env=env)
    assert (process.returncode == 0) == success, (process.args, process.stdout, process.stderr)
    return process


with tempfile.TemporaryDirectory(prefix="uni20-run-examples-") as directory:
    root = Path(directory)
    run("output_live_example", root)
    document = json.loads((root / "results.json").read_text(), parse_float=Decimal)
    assert document["status"] == "partial"
    assert len(document["tables"][0]["data"]["rows"]) == 4
    rows = list(csv.reader(io.StringIO((root / "results.tsv").read_text()), delimiter="\t"))
    assert len(rows) == 5 and rows[1][2] == ""  # Missing residual remains an empty field.
    saved = (root / "results.tsv").read_bytes()
    run("output_live_example", root, success=False)
    assert (root / "results.tsv").read_bytes() == saved  # Exclusive open cannot truncate an existing result.

    named = json.loads(run("output_named_tables_example").stdout, parse_float=Decimal)
    assert [table["name"] for table in named["tables"]] == ["spectrum", "diagnostics"]
    assert named["tables"][0]["data"]["rows"][0][0] == Decimal("0.5")
    replay = json.loads(run("output_replay_example").stdout)
    assert replay["rows"] == [[0], [1], [2]]
    future = json.loads(run("output_no_retention_example").stdout)
    assert future["rows"] == [[1], [2]] and future["first_row"] == 1
    failed = run("output_failure_example")
    assert list(csv.reader(io.StringIO(failed.stdout), delimiter="\t")) == [["answer"], ["42"]]
    assert "Disabled:" in failed.stderr

    state = root / "state.ini"
    state.write_text('U = "3.000000000000000001"\nSpin = "3/2"\n')
    config = root / "job.toml"
    config.write_text('points = 1\nprecision = "long-double"\n[model]\nU = "4.000000000000000001"\n')
    environment = dict(os.environ, UNI20_EXAMPLE_U="invalid", UNI20_EXAMPLE_POINTS="2")
    base = ["--input", state, "--config", config, "--format=named-json"]
    chosen = json.loads(run("run_cli_example", *base, env=environment).stdout, parse_float=Decimal)
    row = chosen["tables"][0]["data"]["rows"][0]
    assert row[1] == Decimal("1.5")
    # Long double may equal double on a target platform; require its exact round-trip text to start with 4.
    assert Decimal(row[2]) >= Decimal("4") and Decimal(row[2]) < Decimal("4.00000000000001")
    explicit = json.loads(run("run_cli_example", *base, "--U=5", env=environment).stdout)
    assert Decimal(explicit["tables"][0]["data"]["rows"][0][2]) == 5
    attributes = json.loads(run("run_cli_example", "--input", state, "--format=named-json", env=environment).stdout)
    assert Decimal(attributes["tables"][0]["data"]["rows"][0][2]) >= 3
    missing = root / "missing.toml"
    would_create = root / "must-not-exist.tsv"
    run("run_cli_example", "--help", "--config", missing, "--output", would_create, env=environment)
    assert not would_create.exists()
    run("run_cli_example", "--config", missing, "--output", would_create, success=False)
    assert not would_create.exists()
    quiet_path = root / "quiet.json"
    quiet = run("run_cli_example", "--quiet", "--format=named-json", "--output", quiet_path)
    assert quiet.stdout == "" and json.loads(quiet_path.read_text())["status"] == "success"
    plain = run("run_cli_example", "--format=commented-tsv", "--no-preamble")
    assert plain.stdout.startswith("point\tspin\tenergy\n") and "#" not in plain.stdout
    # Help callbacks take priority over semantic validation and filesystem access.
    run("run_cli_example", "--help", "--points=-1", "--config", missing)

print("Export, replay, configuration and filesystem contracts passed")
