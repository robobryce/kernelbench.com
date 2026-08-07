from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest


REPO = Path(__file__).resolve().parents[2]
WORKERS = ("brev_worker.sh", "lambda_worker.sh")
REGRADERS = tuple(
    REPO / "benchmarks" / bench / "scripts" / "regrade_sequential.sh"
    for bench in ("hard", "cuda", "mega", "mini")
)


def _worker_env(tmp_path: Path) -> dict[str, str]:
    home = tmp_path / "home"
    home.mkdir(exist_ok=True)
    return os.environ | {
        "HOME": str(home),
        "XDG_STATE_HOME": str(tmp_path / "state"),
        "LAMBDA_API_KEY": "test-only",
    }


@pytest.mark.parametrize("worker", WORKERS)
def test_remote_regrade_rejects_shell_metacharacters_in_run_id(
    tmp_path: Path, worker: str
) -> None:
    completed = subprocess.run(
        [
            str(REPO / "scripts" / worker),
            "regrade",
            "test-worker",
            "run;touch-pwned",
            str(tmp_path / "runs"),
        ],
        text=True,
        capture_output=True,
        check=False,
        env=_worker_env(tmp_path),
        timeout=5,
    )

    assert completed.returncode == 3
    assert "unsafe run_id for remote regrade" in completed.stderr


@pytest.mark.parametrize("worker", WORKERS)
def test_remote_regrade_rejects_unsafe_problem_metadata(
    tmp_path: Path, worker: str
) -> None:
    run = tmp_path / "runs" / "safe-run-id"
    run.mkdir(parents=True)
    (run / "result.json").write_text(
        json.dumps({"problem": "02_safe;touch-pwned"}), encoding="utf-8"
    )
    (run / "solution.py").write_text("pass\n", encoding="utf-8")

    completed = subprocess.run(
        [
            str(REPO / "scripts" / worker),
            "regrade",
            "test-worker",
            run.name,
            str(run.parent),
        ],
        text=True,
        capture_output=True,
        check=False,
        env=_worker_env(tmp_path),
        timeout=5,
    )

    assert completed.returncode == 3
    assert "unsafe problem for remote regrade" in completed.stderr


@pytest.mark.parametrize("regrader", REGRADERS, ids=lambda path: path.parts[-3])
def test_sequential_regrader_prepends_pinned_cuda_bin(regrader: Path) -> None:
    script = regrader.read_text(encoding="utf-8")
    cuda_home = 'export CUDA_HOME="$KBH_CUDA_HOME"'
    cuda_path = 'export PATH="$CUDA_HOME/bin:$PATH"'

    assert cuda_home in script
    assert cuda_path in script
    assert script.index(cuda_home) < script.index(cuda_path)
