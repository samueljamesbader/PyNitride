"""Smoke tests: verify each example script runs to completion without error.

MPLBACKEND=Agg is injected into the subprocess environment so that plt.show()
calls are no-ops, preventing GUI windows from blocking the test. This env var
is child-process-only and does not affect interactive runs.
"""
import os
import subprocess
import sys
from pathlib import Path

import pytest

EXAMPLES_DIR = (
    Path(__file__).parent.parent.parent / "src" / "pynitride" / "examples"
)


def _collect_scripts():
    return sorted(
        p for p in EXAMPLES_DIR.rglob("*.py")
        if p.name != "__init__.py"
    )


@pytest.mark.parametrize(
    "script",
    _collect_scripts(),
    ids=lambda p: p.relative_to(EXAMPLES_DIR).as_posix(),
)
def test_example_runs(script):
    env = {**os.environ, "MPLBACKEND": "Agg"}
    result = subprocess.run(
        [sys.executable, str(script)],
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert result.returncode == 0, (
        f"Script {script.name} failed:\n"
        f"--- stdout ---\n{result.stdout}\n"
        f"--- stderr ---\n{result.stderr}"
    )
