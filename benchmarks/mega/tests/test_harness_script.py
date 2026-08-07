from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUN_HARD = ROOT / "scripts" / "run_hard.sh"
LAUNCH_PARALLEL = ROOT / "scripts" / "launch_parallel_sweep.sh"
RUN_BASELINES = ROOT / "scripts" / "run_baselines.sh"
CLASSIFICATION = ROOT / "src" / "harness" / "classification.py"


def test_post_run_timeout_starts_inside_gpu_lock() -> None:
    script = RUN_HARD.read_text()
    assert "run_gpu_locked_timeout check.py" in script
    assert "run_gpu_locked_timeout benchmark.py" in script
    assert "timeout 180 uv run python check.py" not in script
    assert "timeout 1800 uv run python benchmark.py" not in script


def test_official_grading_uses_immutable_offline_replays() -> None:
    script = RUN_HARD.read_text()
    assert "run_submission_bundle capture" in script
    assert "restore_candidate_from_bundle check" in script
    assert "restore_candidate_from_bundle benchmark" in script
    assert "run_submission_bundle project" in script
    assert script.count("restore_trusted_entrypoint") >= 3
    assert '"${GRADE_COMMAND[@]}" "$TRUSTED_PYTHON" -I' in script
    assert '/usr/bin/mount -o remount,bind,ro "$home"' in script
    assert "--user --map-root-user --net --mount --pid --fork" in script
    assert "--kill-child=KILL --mount-proc" in script


def test_grading_lock_and_helper_are_not_candidate_writable() -> None:
    script = RUN_HARD.read_text()
    assert 'GPU_LOCK_EXEC_SOURCE="$(<"$LOCK_WRAPPER_DIR/gpu-lock-exec")"' in script
    assert '"$real" "$@" 3>&- 9>&-' in script
    assert 'PATH=/usr/bin:/bin /bin/bash -c "$GPU_LOCK_EXEC_SOURCE"' in script
    assert 'TRUSTED_ENTRYPOINT_B64="$(/usr/bin/base64 -w0' in script


def test_submission_bundle_helper_supports_monorepo_and_thin_worker_layouts() -> None:
    script = RUN_HARD.read_text()
    start = script.index('SUBMISSION_BUNDLE_TOOL=""')
    end = script.index('SUBMISSION_BUNDLE_SOURCE=', start)
    resolver = script[start:end]

    monorepo = '"$MONOREPO_ROOT/scripts/lib/submission_bundle.py"'
    bench_local = '"$REPO_ROOT/scripts/lib/submission_bundle.py"'
    assert resolver.index(monorepo) < resolver.index(bench_local)
    assert 'if [ -f "$candidate" ]; then' in resolver
    assert 'if [ -z "$SUBMISSION_BUNDLE_TOOL" ]; then' in resolver


def test_baseline_generator_opts_into_reference_diagnostics() -> None:
    script = RUN_BASELINES.read_text()
    assert "KBH_BENCHMARK_BASELINES=1 timeout 300 uv run python benchmark.py" in script


def test_run_archives_are_allocated_atomically() -> None:
    script = RUN_HARD.read_text()
    assert 'RUN_DIR_BASE="${REPO_ROOT}/outputs/runs/' in script
    assert 'if mkdir "$candidate" 2>/dev/null; then' in script
    assert 'failed to allocate unique run directory' in script


def test_claude_family_runs_from_archive_workspace() -> None:
    script = RUN_HARD.read_text()
    for harness in ("claude)", "ccr-claude)", "zai-claude)", "minimax-claude)"):
        start = script.index(harness)
        end = script.index(";;", start)
        block = script[start:end]
        assert '( cd "$PROBLEM_DIR" &&' in block
        assert 'timeout "$BUDGET_SECONDS"' in block
        assert '--add-dir "$PROBLEM_DIR"' in block


def test_minimax_claude_uses_official_anthropic_endpoint() -> None:
    script = RUN_HARD.read_text()
    start = script.index("minimax-claude)")
    end = script.index(";;", start)
    block = script[start:end]
    assert "MINIMAX_API_KEY" in block
    assert "https://api.minimax.io/anthropic" in block
    assert 'ANTHROPIC_MODEL="$MODEL"' in block
    assert 'ANTHROPIC_DEFAULT_OPUS_MODEL="$MODEL"' in block
    assert 'env \\' not in block


def test_check_timeouts_are_retryable_not_plain_check_failed() -> None:
    classifier = CLASSIFICATION.read_text()
    assert 'reason = "check_timeout"' in classifier
    assert 'reason = "benchmark_timeout"' in classifier
    assert "elif check_exit == 124:" in classifier


def test_grok_uses_headless_cli_and_end_marker() -> None:
    script = RUN_HARD.read_text()
    start = script.index("grok)")
    end = script.index(";;", start)
    block = script[start:end]
    assert 'timeout "$BUDGET_SECONDS" grok' in block
    assert '--cwd "$PROBLEM_DIR"' in block
    assert "--output-format streaming-json" in block
    assert '"type":"end"' in script


def test_parallel_launcher_keeps_run_hard_jobs_waitable() -> None:
    script = LAUNCH_PARALLEL.read_text()
    assert 'LAST_LAUNCH_PID=$!' in script
    assert 'pid="$(launch_one' not in script
    assert 'launch_one "$name" "$harness" "$model" "$effort" "$problem"' in script
    assert 'pid="$LAST_LAUNCH_PID"' in script
    assert 'wait "$pid" || true' in script


def test_cuda_cannot_be_disabled_for_agent_phase() -> None:
    script = RUN_HARD.read_text()
    parallel = LAUNCH_PARALLEL.read_text()
    retries = (ROOT / "scripts" / "launch_infra_retries.sh").read_text()
    for text in (script, parallel, retries):
        assert "KBH_DISABLE_AGENT_CUDA" not in text
        assert "AGENT_CUDA_ENV" not in text
        assert "KBH_AGENT_PHASE" not in text
        assert "CUDA_VISIBLE_DEVICES=" not in text
