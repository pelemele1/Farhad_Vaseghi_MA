import shutil
import subprocess

import pytest

_BASH = shutil.which("bash")

pytestmark = pytest.mark.skipif(_BASH is None, reason="requires a bash executable on PATH")


@pytest.mark.parametrize("script", [
    "scripts/hpc/setup_env.sh",
    "scripts/hpc/train_stage_a.slurm",
    "scripts/hpc/train_stage_b.slurm",
])
def test_hpc_script_has_valid_bash_syntax(script):
    result = subprocess.run([_BASH, "-n", script], capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
