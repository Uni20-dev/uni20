"""Prove source identity comes from the selected tree, including overrides and archive builds."""
from pathlib import Path
import subprocess
import sys
import tempfile

cmake, script = sys.argv[1:]

with tempfile.TemporaryDirectory(prefix="uni20-provenance-") as directory:
    root = Path(directory)
    source = root / "source"
    source.mkdir()
    output = root / "identity.hpp"

    def generate(path=source, override=""):
        subprocess.run([cmake, f"-DSOURCE_DIR={path}", f"-DOUTPUT={output}", "-DNAMESPACE=test_build",
                        f"-DREVISION_OVERRIDE={override}", "-P", script], check=True, capture_output=True)
        return output.read_text()

    assert '"unknown"' in generate()
    assert 'release-1\\"archive' in generate(override='release-1"archive')
    subprocess.run(["git", "init", "-q", str(source)], check=True)
    (source / "CMakeLists.txt").write_text("# fixture\n")
    subprocess.run(["git", "-C", str(source), "add", "CMakeLists.txt"], check=True)
    subprocess.run(["git", "-C", str(source), "-c", "user.name=Uni20 test", "-c", "user.email=test@example.invalid",
                    "-c", "commit.gpgsign=false", "commit", "-qm", "fixture"], check=True)
    head = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    assert f'"{head}"' in generate()
    (source / "CMakeLists.txt").write_text("# changed\n")
    assert f'"{head}-dirty"' in generate()
    archive = source / "untracked-archive"
    archive.mkdir()
    (archive / "CMakeLists.txt").write_text("# independent archive\n")
    assert '"unknown"' in generate(archive)  # Do not inherit the unrelated parent repository identity.

print("Source identity, dirty tree, archive and explicit override contracts passed")
