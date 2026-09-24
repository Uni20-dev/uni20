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
    for name in ("run_metadata_example", "configuration_sources_example"):
        assert run(name).stdout.startswith("Example:")
    live = run("output_live_example", root)
    assert live.stdout.startswith("Example:") and "residual" in live.stdout
    document = json.loads((root / "results.json").read_text(), parse_float=Decimal)
    assert document["status"] == "complete" and document["summary"]["outcome"] == "partial"
    assert document["tables"][0]["data"]["summary"]["converged"] == "false"
    assert len(document["tables"][0]["data"]["rows"]) == 4
    rows = list(csv.reader(io.StringIO((root / "results.tsv").read_text()), delimiter="\t"))
    assert len(rows) == 5 and rows[1][2] == ""  # Missing residual remains an empty field.
    saved = (root / "results.tsv").read_bytes()
    run("output_live_example", root, success=False)
    assert (root / "results.tsv").read_bytes() == saved  # Exclusive open cannot truncate an existing result.

    # Explain exports on stderr without contaminating machine-readable stdout.
    named_output = run("output_named_tables_example")
    assert named_output.stderr.startswith("Example:")
    named = json.loads(named_output.stdout, parse_float=Decimal)
    assert [table["name"] for table in named["tables"]] == ["spectrum", "diagnostics"]
    assert named["tables"][0]["data"]["rows"][0][0] == Decimal("0.5")
    replay_output = run("output_replay_example")
    assert replay_output.stderr.startswith("Example:")
    replay = json.loads(replay_output.stdout)
    assert replay["rows"] == [[0], [1], [2]]
    future_output = run("output_no_retention_example")
    assert future_output.stderr.startswith("Example:")
    future = json.loads(future_output.stdout)
    assert future["rows"] == [[1], [2]] and future["first_row"] == 1
    failed = run("output_failure_example")
    assert list(csv.reader(io.StringIO(failed.stdout), delimiter="\t")) == [["answer"], ["42"]]
    assert failed.stderr.startswith("Example:") and "Disabled:" in failed.stderr

    state = root / "state.ini"
    state.write_text('U = "3.000000000000000001"\nSpin = "3/2"\n')
    config = root / "job.toml"
    config.write_text('points = 1\nprecision = "long-double"\n[model]\nU = "4.000000000000000001"\n')
    environment = dict(os.environ, UNI20_EXAMPLE_U="invalid", UNI20_EXAMPLE_POINTS="2")
    base = ["--input", state, "--config", config, "--format=named-json"]
    chosen_output = run("run_cli_example", *base, env=environment)
    assert chosen_output.stderr.startswith("Example:")
    chosen = json.loads(chosen_output.stdout, parse_float=Decimal)
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
    assert quiet.stdout == "" and json.loads(quiet_path.read_text())["status"] == "complete"
    assert quiet.stderr == ""
    plain = run("run_cli_example", "--format=commented-tsv", "--no-preamble")
    assert plain.stdout.startswith("point\tspin\tenergy\n") and "#" not in plain.stdout
    assert plain.stderr == ""
    terminal = run("run_cli_example", "--points=1")
    assert terminal.stdout.startswith("Example:") and terminal.stderr == ""
    for format, delimiter in (("commented-csv", ","), ("commented-tsv", "\t")):
        comments = run("run_cli_example", "--points=1", f"--format={format}")
        assert "# outcome: success\n" in comments.stdout and "# program: run_cli_example\n" in comments.stdout
        uncommented = "".join(line for line in comments.stdout.splitlines(keepends=True) if not line.startswith("#"))
        assert list(csv.reader(io.StringIO(uncommented), delimiter=delimiter)) == [["point", "spin", "energy"], ["0", "0.5", "4"]]
        plain = run("run_cli_example", "--points=1", f"--format={format}", "--no-preamble")
        assert plain.stdout == uncommented and plain.stderr == ""

    # Independent terminal/stdout format plus repeatable exports; preserve colons and spaces in paths.
    csv_path = root / "screen export.csv"
    json_path = root / "screen:export.json"
    screen = run("run_cli_example", "--points=2", f"--export=csv:{csv_path}", f"--export=named-json:{json_path}")
    assert "Illustrative dispersion" in screen.stdout and screen.stderr == ""
    assert list(csv.reader(io.StringIO(csv_path.read_text()))) == [["point", "spin", "energy"], ["0", "0.5", "4"], ["1", "0.5", "2"]]
    assert json.loads(json_path.read_text())["tables"][0]["data"]["rows"] == [[0, 0.5, "4"], [1, 0.5, "2"]]
    saved_csv, saved_json = csv_path.read_bytes(), json_path.read_bytes()
    run("run_cli_example", "--quiet", f"--export=csv:{csv_path}", f"--export=named-json:{json_path}", success=False)
    assert csv_path.read_bytes() == saved_csv and json_path.read_bytes() == saved_json
    machine = run("run_cli_example", "--format=json", "--overwrite", f"--export=csv:{csv_path}")
    assert len(json.loads(machine.stdout)["rows"]) == 4
    quiet = run("run_cli_example", "--quiet", "--overwrite", f"--export=csv:{csv_path}", f"--export=named-json:{json_path}")
    assert quiet.stdout == quiet.stderr == ""
    assert json.loads(json_path.read_text())["summary"]["outcome"] == "success"
    # --output remains a primary file destination and can be combined with extra exports.
    primary = run("run_cli_example", "--quiet", "--overwrite", "--format=csv", "--output", csv_path,
                  f"--export=named-json:{json_path}")
    assert primary.stdout == primary.stderr == ""
    alias = root / "alias.csv"
    run("run_cli_example", f"--export=csv:{alias}", f"--export=tsv:{alias}", success=False)
    assert not alias.exists()
    for bad in ("csv", "csv:", "unknown:file"):
        run("run_cli_example", f"--export=csv:{would_create}", f"--export={bad}", success=False)
        assert not would_create.exists()
    run("run_cli_example", "--help", f"--export=csv:{would_create}")
    assert not would_create.exists()
    # Help callbacks take priority over semantic validation and filesystem access.
    run("run_cli_example", "--help", "--points=-1", "--config", missing)

print("Export, replay, configuration and filesystem contracts passed")
