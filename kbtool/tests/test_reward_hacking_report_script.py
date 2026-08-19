import hashlib
import os
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "scripts/reward_hacking_report.sh"


def _write_executable(path: Path, contents: str) -> None:
    path.write_text(contents)
    path.chmod(0o755)


def _fake_tools(tmp_path: Path) -> tuple[dict[str, str], Path]:
    tools = tmp_path / "bin"
    tools.mkdir()
    log = tmp_path / "autocuda.log"
    _write_executable(
        tools / "autocuda",
        """#!/usr/bin/env bash
set -euo pipefail
printf '%s\\n' "$*" >> "$AUTOCUDA_LOG"
if [[ $1 == report && $2 == html ]]; then
    printf '<!DOCTYPE html><title>report</title>\\n' > "${3%.md}.html"
fi
""",
    )
    _write_executable(tools / "pandoc", "#!/usr/bin/env bash\nexit 0\n")
    env = os.environ.copy()
    env["PATH"] = f"{tools}:{env['PATH']}"
    env["AUTOCUDA_LOG"] = str(log)
    return env, log


def _run(*args: str | Path, env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [str(SCRIPT), *(str(arg) for arg in args)],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )


def test_render_validates_schema_before_writing_html(tmp_path: Path) -> None:
    env, log = _fake_tools(tmp_path)
    report = tmp_path / "sample-report-reward-hacking.md"
    report.write_text("# Sample\n")

    result = _run("render", report, env=env)

    assert result.returncode == 0, result.stderr
    assert report.with_suffix(".html").is_file()
    calls = log.read_text().splitlines()
    assert calls[0].startswith("schema check report-reward-hacking-markdown")
    assert f"--data-dir {tmp_path}" in calls[0]
    assert calls[0].endswith("--output sample")
    assert calls[1] == f"report html {report}"


def test_verify_checks_schema_html_and_hashes(tmp_path: Path) -> None:
    env, _ = _fake_tools(tmp_path)
    report = tmp_path / "sample-report-reward-hacking.md"
    html = tmp_path / "sample-report-reward-hacking.html"
    report.write_text("# Sample\n")
    html.write_text("<!DOCTYPE html><title>sample</title>\n")
    checksums = tmp_path / "SHA256SUMS"
    checksums.write_text(
        "\n".join(
            f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.name}"
            for path in (report, html)
        )
        + "\n"
    )

    result = _run("verify", report, env=env)

    assert result.returncode == 0, result.stderr
    assert "sample-report-reward-hacking.md: OK" in result.stdout
    assert "sample-report-reward-hacking.html: OK" in result.stdout


def test_rejects_noncanonical_report_name(tmp_path: Path) -> None:
    env, _ = _fake_tools(tmp_path)
    report = tmp_path / "report.md"
    report.write_text("# Sample\n")

    result = _run("render", report, env=env)

    assert result.returncode == 1
    assert "must end with -report-reward-hacking.md" in result.stderr
