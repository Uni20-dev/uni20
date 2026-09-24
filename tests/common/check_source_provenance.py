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
    subprocess.run(["git", "-C", str(source), "add", "untracked-archive/CMakeLists.txt"], check=True)
    # Vendoring the archive in the consumer repository must not turn its HEAD into Uni20's identity.
    assert '"unknown"' in generate(archive)
    assert '"release-archive"' in generate(archive, override="release-archive")

    linked = root / "linked-worktree"
    subprocess.run(["git", "-C", str(source), "worktree", "add", "-q", "--detach", str(linked), "HEAD"], check=True)
    assert f'"{head}"' in generate(linked)  # Worktrees use a .git file, not a .git directory.
    alias = root / "source-alias"
    alias.symlink_to(linked, target_is_directory=True)
    assert f'"{head}"' in generate(alias)

print("Source identity, dirty tree, vendored archives, worktrees, symlinks and explicit override contracts passed")
