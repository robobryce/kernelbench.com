import ast
import importlib.util
import os
import posix
import shutil
import subprocess
import sys
import types
from pathlib import Path

import pytest

BENCH_ROOT = Path(__file__).resolve().parents[1]
MONOREPO = BENCH_ROOT.parents[1]
ENTRYPOINT = BENCH_ROOT / "src" / "eval" / "trusted_entrypoint.py"


def test_print_pass_then_system_exit_zero_is_not_success(tmp_path: Path) -> None:
    (tmp_path / "check.py").write_text("import solution\nprint('PASS')\n")
    (tmp_path / "solution.py").write_text("print('PASS')\nraise SystemExit(0)\n")

    completed = subprocess.run(
        [sys.executable, str(ENTRYPOINT), "check.py"],
        cwd=tmp_path,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode != 0
    assert completed.stdout.splitlines().count("PASS") == 1
    assert "before normal completion" in completed.stderr


def test_trusted_entrypoint_preserves_nonzero_system_exit(tmp_path: Path) -> None:
    (tmp_path / "check.py").write_text("raise SystemExit(7)\n")

    completed = subprocess.run(
        [sys.executable, str(ENTRYPOINT), "check.py"],
        cwd=tmp_path,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 7


def test_solution_cannot_rebind_grading_script_globals(tmp_path: Path) -> None:
    (tmp_path / "check.py").write_text(
        "def property_guard():\n"
        "    print('PROPERTY_GUARD')\n"
        "import solution\n"
        "property_guard()\n"
        "print('PASS')\n"
    )
    (tmp_path / "solution.py").write_text(
        "import __main__\n__main__.property_guard = lambda: None\n"
    )

    completed = subprocess.run(
        [sys.executable, str(ENTRYPOINT), "check.py"],
        cwd=tmp_path,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode != 0
    assert "PASS" not in completed.stdout
    assert "import __main__" in completed.stderr


def test_solution_cannot_rebind_entrypoint_exit_handling(tmp_path: Path) -> None:
    (tmp_path / "check.py").write_text("import solution\nprint('PASS')\n")
    (tmp_path / "solution.py").write_text(
        "import __main__\n"
        "__main__.SystemExit = RuntimeError\n"
        "__main__.print = lambda *_args, **_kwargs: (_ for _ in ()).throw(SystemExit(0))\n"
        "raise SystemExit(0)\n"
    )

    completed = subprocess.run(
        [sys.executable, str(ENTRYPOINT), "check.py"],
        cwd=tmp_path,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode != 0
    assert "PASS" not in completed.stdout
    assert "import __main__" in completed.stderr


@pytest.mark.parametrize("module", ["os", "posix"])
def test_process_exit_cannot_bypass_grading(tmp_path: Path, module: str) -> None:
    (tmp_path / "check.py").write_text("import solution\nprint('PASS')\n")
    (tmp_path / "solution.py").write_text(f"import {module}\n{module}._exit(0)\n")

    completed = subprocess.run(
        [sys.executable, str(ENTRYPOINT), "check.py"],
        cwd=tmp_path,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode != 0
    assert "PASS" not in completed.stdout
    assert "termination or replacement" in completed.stderr


def test_process_guards_are_restored_after_grading(tmp_path: Path) -> None:
    spec = importlib.util.spec_from_file_location("test_trusted_entrypoint", ENTRYPOINT)
    assert spec and spec.loader
    entrypoint = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(entrypoint)
    originals = {(os, name): getattr(os, name) for name in ("_exit", "execl", "execv", "spawnv")}
    originals[(posix, "_exit")] = posix._exit
    (tmp_path / "check.py").write_text("import solution\nprint('PASS')\n")
    (tmp_path / "solution.py").write_text("VALUE = 1\n")

    assert entrypoint.run_grading_script(tmp_path / "check.py") == 0
    assert all(getattr(module, name) is value for (module, name), value in originals.items())


def test_torch_stack_is_not_rejected_as_frame_introspection(tmp_path: Path) -> None:
    (tmp_path / "check.py").write_text(
        "import solution\nassert solution.RESULT == [[1, 2], [3, 4]]\nprint('PASS')\n"
    )
    (tmp_path / "solution.py").write_text(
        "import torch\n"
        "RESULT = torch.stack((torch.tensor([1, 2]), torch.tensor([3, 4]))).tolist()\n"
    )

    completed = subprocess.run(
        [sys.executable, str(ENTRYPOINT), "check.py"],
        cwd=tmp_path,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.splitlines() == ["PASS"]


@pytest.mark.parametrize(
    "source",
    [
        "import inspect\ninspect.stack()\n",
        "import inspect as frames\nframes.stack()\n",
        "from inspect import stack as frames\nframes()\n",
        "import inspect\ngetattr(inspect, 'stack')()\n",
    ],
    ids=["attribute", "module-alias", "function-alias", "dynamic-attribute"],
)
def test_inspect_stack_remains_a_checker_control_tripwire(tmp_path: Path, source: str) -> None:
    (tmp_path / "check.py").write_text("import solution\nprint('PASS')\n")
    (tmp_path / "solution.py").write_text(source)

    completed = subprocess.run(
        [sys.executable, str(ENTRYPOINT), "check.py"],
        cwd=tmp_path,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode != 0
    assert "PASS" not in completed.stdout
    assert "inspect.stack" in completed.stderr


@pytest.mark.parametrize(
    "source",
    [
        "import os\nprint('PASS', flush=True)\nos.execv('/bin/true', ['true'])\n",
        "import ctypes\nprint('PASS', flush=True)\nctypes.CDLL(None).syscall(231, 0)\n",
        "import ctypes\nprint('PASS', flush=True)\nctypes.CDLL(None).exit(0)\n",
        "import ctypes\ngetattr(ctypes.CDLL(None), 'ex' + 'it')(0)\n",
        "import sys\nframe = sys._getframe()\nframe.f_back.f_locals.clear()\n",
        "import os\nos.chdir('decoy')\n",
        "import sys\nsys.modules['shapes'].SHAPES.clear()\n",
    ],
    ids=[
        "exec",
        "native-syscall",
        "native-exit",
        "obfuscated-native-exit",
        "frame-walk",
        "chdir",
        "module-poison",
    ],
)
def test_candidate_control_flow_primitives_are_rejected_before_execution(
    tmp_path: Path, source: str
) -> None:
    (tmp_path / "check.py").write_text("import solution\nprint('PASS')\n")
    (tmp_path / "solution.py").write_text(source)

    completed = subprocess.run(
        [sys.executable, str(ENTRYPOINT), "check.py"],
        cwd=tmp_path,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode != 0
    assert "PASS" not in completed.stdout
    assert "forbidden" in completed.stderr


def test_solution_package_cannot_shadow_solution_file(tmp_path: Path) -> None:
    (tmp_path / "check.py").write_text("import solution\nprint('PASS')\n")
    (tmp_path / "solution.py").write_text("VALUE = 'reviewed'\n")
    package = tmp_path / "solution"
    package.mkdir()
    (package / "__init__.py").write_text("print('PASS')\n")

    completed = subprocess.run(
        [sys.executable, str(ENTRYPOINT), "check.py"],
        cwd=tmp_path,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode != 0
    assert "PASS" not in completed.stdout
    assert "shadows trusted module" in completed.stderr


def test_precompiled_binary_is_rejected_regardless_of_suffix(tmp_path: Path) -> None:
    (tmp_path / "check.py").write_text("import solution\nprint('PASS')\n")
    (tmp_path / "solution.py").write_text("VALUE = 1\n")
    (tmp_path / "payload.bin").write_bytes(b"\x7fELF\x02\x01payload")

    completed = subprocess.run(
        [sys.executable, str(ENTRYPOINT), "check.py"],
        cwd=tmp_path,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode != 0
    assert "PASS" not in completed.stdout
    assert "precompiled candidate binary" in completed.stderr


def test_isolated_replay_bootstrap_uses_system_exit_guard(tmp_path: Path) -> None:
    shared = (MONOREPO / "scripts" / "lib" / "run_harness.sh").read_text()
    replay_source = shared.split("SUBMISSION_REPLAY_SOURCE='", 1)[1].split("\n'\nif !", 1)[0]
    problem = tmp_path / "repo" / "problems" / "p"
    problem.mkdir(parents=True)
    (problem / "check.py").write_text("import solution\nprint('PASS')\n")
    (problem / "solution.py").write_text("print('PASS')\nraise SystemExit(0)\n")

    completed = subprocess.run(
        [
            sys.executable,
            "-I",
            "-c",
            replay_source,
            str(ENTRYPOINT),
            "check.py",
        ],
        cwd=problem,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode != 0
    assert completed.stdout.splitlines().count("PASS") == 1
    assert "before normal completion" in completed.stderr


def test_native_runner_uses_the_system_exit_guard(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    native = _load_native_harness(monkeypatch)
    helper = tmp_path / "src" / "eval" / "trusted_entrypoint.py"
    helper.parent.mkdir(parents=True)
    shutil.copy2(ENTRYPOINT, helper)
    problem = tmp_path / "problems" / "p"
    problem.mkdir(parents=True)
    (problem / "check.py").write_text("import solution\nprint('PASS')\n")
    (problem / "solution.py").write_text("print('PASS')\nraise SystemExit(0)\n")
    native._ALLOW_UNISOLATED_NATIVE_FOR_TESTS = True

    returncode, log = native.run_native(str(tmp_path), "p", "check.py", 5)

    assert returncode != 0
    assert log.splitlines().count("PASS") == 1


def test_native_workspace_restores_trusted_files_and_removes_shadow_packages(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    native = _load_native_harness(monkeypatch)
    bench = tmp_path / "bench"
    helper = bench / "src" / "eval" / "trusted_entrypoint.py"
    helper.parent.mkdir(parents=True)
    helper.write_text("TRUSTED = True\n")
    source_problem = bench / "problems" / "p"
    source_problem.mkdir(parents=True)
    (source_problem / "check.py").write_text("print('PASS')\n")
    (source_problem / "benchmark.py").write_text("print('peak_fraction: 1')\n")
    (source_problem / "reference.py").write_text("TRUSTED = True\n")
    (source_problem / "shapes.py").write_text("SHAPES = [1]\n")

    workspace = Path(native.make_workspace(str(bench), "p"))
    problem = workspace / "problems" / "p"
    try:
        (workspace / "src" / "eval" / "trusted_entrypoint.py").write_text("POISONED = True\n")
        (problem / "check.py").write_text("print('PASS')\nprint('PASS')\n")
        (problem / "shapes").mkdir()
        (problem / "shapes" / "__init__.py").write_text("SHAPES = []\n")

        native._restore_trusted_workspace(str(workspace), "p")

        assert (workspace / "src" / "eval" / "trusted_entrypoint.py").read_text() == (
            "TRUSTED = True\n"
        )
        assert (problem / "check.py").read_text() == "print('PASS')\n"
        assert not (problem / "shapes").exists()
    finally:
        native._TRUSTED_WORKSPACE_FILES.pop(str(workspace.resolve()), None)
        shutil.rmtree(workspace, ignore_errors=True)


def test_native_runner_restores_candidate_after_each_subprocess(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    native = _load_native_harness(monkeypatch)
    bench = tmp_path / "bench"
    helper = bench / "src" / "eval" / "trusted_entrypoint.py"
    helper.parent.mkdir(parents=True)
    shutil.copy2(ENTRYPOINT, helper)
    source_problem = bench / "problems" / "p"
    source_problem.mkdir(parents=True)
    (source_problem / "check.py").write_text(
        "from pathlib import Path\n"
        "Path('solution.py').write_text('POISONED = True\\n')\n"
        "Path('created.py').write_text('POISONED = True\\n')\n"
        "print('PASS')\n"
    )
    (source_problem / "benchmark.py").write_text("print('peak_fraction: 1')\n")

    workspace = Path(native.make_workspace(str(bench), "p"))
    problem = workspace / "problems" / "p"
    original = b"ORIGINAL = True\n"
    (problem / "solution.py").write_bytes(original)
    try:
        native._ALLOW_UNISOLATED_NATIVE_FOR_TESTS = True
        returncode, log = native.run_native(str(workspace), "p", "check.py", 5)

        assert returncode == 0, log
        assert (problem / "solution.py").read_bytes() == original
        assert not (problem / "created.py").exists()
    finally:
        native._TRUSTED_WORKSPACE_FILES.pop(str(workspace.resolve()), None)
        shutil.rmtree(workspace, ignore_errors=True)


def test_native_workspace_rejects_symlinked_intermediate_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    native = _load_native_harness(monkeypatch)
    bench = tmp_path / "bench"
    helper = bench / "src" / "eval" / "trusted_entrypoint.py"
    helper.parent.mkdir(parents=True)
    helper.write_text("TRUSTED = True\n")
    source_problem = bench / "problems" / "p"
    source_problem.mkdir(parents=True)
    (source_problem / "check.py").write_text("print('PASS')\n")

    workspace = Path(native.make_workspace(str(bench), "p"))
    outside = tmp_path / "outside-problems"
    try:
        (workspace / "problems").rename(outside)
        (workspace / "problems").symlink_to(outside, target_is_directory=True)

        with pytest.raises(RuntimeError, match="problems workspace was replaced"):
            native._restore_trusted_workspace(str(workspace), "p")
    finally:
        native._TRUSTED_WORKSPACE_FILES.pop(str(workspace.resolve()), None)
        if (workspace / "problems").is_symlink():
            (workspace / "problems").unlink()
        shutil.rmtree(workspace, ignore_errors=True)
        shutil.rmtree(outside, ignore_errors=True)


def test_native_runner_uses_offline_read_only_home_sandbox() -> None:
    source = (MONOREPO / "environments" / "kernel_hard" / "kernel_native_harness.py").read_text()
    assert '"--map-root-user"' in source
    assert '"--net"' in source
    assert '"--pid"' in source
    assert '"--kill-child=KILL"' in source
    assert '/usr/bin/mount -o remount,bind,ro "$home"' in source
    assert 'raise RuntimeError("trusted native grading requires unshare and setpriv")' in source


def test_native_harness_is_mirrored_exactly() -> None:
    hard = MONOREPO / "environments" / "kernel_hard" / "kernel_native_harness.py"
    mega = MONOREPO / "environments" / "kernel_mega" / "kernel_native_harness.py"
    assert hard.read_bytes() == mega.read_bytes()


def test_trusted_entrypoint_is_mirrored_exactly() -> None:
    expected = ENTRYPOINT.read_bytes()
    mirrors = [
        MONOREPO / "benchmarks" / bench / "src" / "eval" / "trusted_entrypoint.py"
        for bench in ("cuda", "mega", "mini")
    ] + [
        MONOREPO / "environments" / env / "_bench" / "src" / "eval" / "trusted_entrypoint.py"
        for env in ("kernel_hard", "kernel_mega")
    ]
    assert all(path.read_bytes() == expected for path in mirrors)


def test_every_checker_scans_candidate_source_before_import() -> None:
    checks = sorted(
        path
        for pattern in (
            "benchmarks/hard/problems-*/*/check.py",
            "benchmarks/cuda/problems-*/*/check.py",
            "benchmarks/mini/problems-*/*/check.py",
            "benchmarks/mega/problems/*/check.py",
            "environments/kernel_hard/_bench/problems/*/check.py",
            "environments/kernel_mega/_bench/problems/*/check.py",
        )
        for path in MONOREPO.glob(pattern)
    )
    assert len(checks) == 43
    for path in checks:
        tree = ast.parse(path.read_bytes(), filename=str(path))
        solution_imports = [
            node.lineno
            for node in ast.walk(tree)
            if isinstance(node, ast.Import)
            and any(alias.name == "solution" for alias in node.names)
        ]
        forbidden_ends = [
            node.end_lineno
            for node in ast.walk(tree)
            if isinstance(node, ast.For)
            and isinstance(node.target, ast.Name)
            and node.target.id == "forbidden"
        ]
        cuda_gate_ends = [
            node.end_lineno
            for node in ast.walk(tree)
            if isinstance(node, ast.If)
            and isinstance(node.test, ast.UnaryOp)
            and isinstance(node.test.op, ast.Not)
            and isinstance(node.test.operand, ast.Name)
            and node.test.operand.id == "ok"
            and node.lineno < solution_imports[0]
        ]
        assert len(solution_imports) == 1, path
        assert forbidden_ends, path
        assert solution_imports[0] > max(forbidden_ends), path
        if cuda_gate_ends:
            assert solution_imports[0] > max(cuda_gate_ends), path


def test_every_benchmark_loads_problem_metadata_before_candidate_import() -> None:
    benchmarks = sorted(
        path
        for pattern in (
            "benchmarks/hard/problems-*/*/benchmark.py",
            "benchmarks/cuda/problems-*/*/benchmark.py",
            "benchmarks/mini/problems-*/*/benchmark.py",
            "benchmarks/mega/problems/*/benchmark.py",
            "environments/kernel_hard/_bench/problems/*/benchmark.py",
            "environments/kernel_mega/_bench/problems/*/benchmark.py",
        )
        for path in MONOREPO.glob(pattern)
    )
    assert len(benchmarks) == 43
    for path in benchmarks:
        tree = ast.parse(path.read_bytes(), filename=str(path))
        solution_imports = [
            node.lineno
            for node in ast.walk(tree)
            if isinstance(node, ast.Import)
            and any(alias.name == "solution" for alias in node.names)
        ]
        metadata_reads = [
            node.lineno
            for node in ast.walk(tree)
            if isinstance(node, ast.Constant) and node.value == "problem.yaml"
        ]
        assert len(solution_imports) == 1, path
        assert metadata_reads, path
        assert max(metadata_reads) < solution_imports[0], path


def _load_native_harness(monkeypatch: pytest.MonkeyPatch):
    fake_vf = types.ModuleType("verifiers")
    fake_vf.StatefulToolEnv = object
    fake_vf.Rubric = object
    fake_vf.Environment = object
    fake_vf.Parser = object
    fake_vf.RubricGroup = object
    fake_datasets = types.ModuleType("datasets")
    fake_datasets.Dataset = object
    monkeypatch.setitem(sys.modules, "verifiers", fake_vf)
    monkeypatch.setitem(sys.modules, "datasets", fake_datasets)

    path = MONOREPO / "environments" / "kernel_hard" / "kernel_native_harness.py"
    spec = importlib.util.spec_from_file_location("test_kernel_native_harness", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    ("runs", "correct", "score"),
    [
        ([(1, "PASS\n")], False, 0.0),
        ([(0, "PASS\nPASS\n")], False, 0.0),
        ([(0, "PASS\n"), (1, "peak_fraction: 0.75\n")], True, 0.0),
        (
            [(0, "PASS\n"), (0, "peak_fraction: 0.75\npeak_fraction: 0.9\n")],
            True,
            0.0,
        ),
        ([(0, "PASS\n"), (0, "peak_fraction: 0.75\n")], True, 0.75),
    ],
)
def test_native_scoring_requires_exit_codes_and_single_markers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    runs: list[tuple[int, str]],
    correct: bool,
    score: float,
) -> None:
    native = _load_native_harness(monkeypatch)
    problem = tmp_path / "problems" / "p"
    problem.mkdir(parents=True)
    (problem / "solution.py").write_text("# candidate\n")
    results = iter(runs)
    monkeypatch.setattr(native, "run_native", lambda *_args: next(results))

    result = native.score_workspace(str(tmp_path), "p", 1, 1)

    assert result["correct"] is correct
    assert result["peak_fraction"] == score


def test_shell_grading_uses_exact_markers_and_process_status() -> None:
    shared = (MONOREPO / "scripts" / "lib" / "run_harness.sh").read_text()
    assert "grep -axc 'PASS'" in shared
    assert '[ "$CHECK_PASS_COUNT" -eq 1 ]' in shared
    assert '[ "$BENCH_EXIT_CODE" -eq 0 ]' in shared
    assert '[ "$BENCH_METRIC_COUNT" -eq 1 ]' in shared
    assert 'trusted_entrypoint="$stage_dir/repo/src/eval/trusted_entrypoint.py"' in shared
    assert '"$trusted_entrypoint" "$script"' in shared
    shared_check = shared.index("run_replay_stage check.py")
    shared_benchmark = shared.index("run_replay_stage benchmark.py")
    assert shared.index('strip_python_bytecode "$PROBLEM_DIR"') < shared_check
    assert shared_check < shared_benchmark
    assert "prepare_replay_stage check" in shared
    assert "prepare_replay_stage benchmark" in shared
    assert 'find "$stage_root" -type d -name __pycache__' in shared

    mega = (BENCH_ROOT.parent / "mega" / "scripts" / "run_hard.sh").read_text()
    assert "grep -axc 'PASS'" in mega
    assert '[ "$BENCH_METRIC_COUNT" -eq 1 ]' in mega
    mega_check = mega.index('"$TRUSTED_ENTRYPOINT" check.py')
    mega_benchmark = mega.index('"$TRUSTED_ENTRYPOINT" benchmark.py')
    assert mega.index('strip_python_bytecode "$PROBLEM_DIR"') < mega_check
    assert mega.index('strip_python_bytecode "$PROBLEM_DIR"', mega_check) < mega_benchmark
