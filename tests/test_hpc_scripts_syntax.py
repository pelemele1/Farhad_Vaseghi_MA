import shutil
import subprocess

import pytest

_BASH = shutil.which("bash")

pytestmark = pytest.mark.skipif(_BASH is None, reason="requires a bash executable on PATH")

CRLF = bytes([13, 10])


@pytest.mark.parametrize("script", [
    "scripts/hpc/setup_env.sh",
    "scripts/hpc/train_stage_a.slurm",
    "scripts/hpc/train_stage_b.slurm",
    "scripts/hpc/train_impaired_gate.slurm",
    "scripts/hpc/train_visible.sh",
    "scripts/hpc/evaluate_visible.sh",
    "scripts/hpc/build_visible_6class.slurm",
    "scripts/hpc/run_tests_parallel.sh",
])
def test_hpc_script_has_valid_bash_syntax(script):
    result = subprocess.run([_BASH, "-n", script], capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr


def test_build_job_has_unix_line_endings():
    # Slurm rejects a batch script with Windows (CRLF) line endings.
    with open("scripts/hpc/build_visible_6class.slurm", "rb") as f:
        assert CRLF not in f.read()
