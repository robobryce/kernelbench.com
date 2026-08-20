"""Deterministic properties and security checks for submission bundles."""

from __future__ import annotations

import json
import os
import random
import shutil
import stat
import subprocess
import sys
from pathlib import Path

import pytest


REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from scripts import kernel_sidecars as sidecars  # noqa: E402
from scripts.lib import submission_bundle as bundles  # noqa: E402


FileTree = dict[str, tuple[bytes, bool]]
ROUND_TRIP_SEEDS = tuple(0xB00D + case for case in range(24))
ORDER_SEEDS = tuple(0xC0DE + case for case in range(12))


def _generated_tree(seed: int) -> FileTree:
    rng = random.Random(seed)
    tree: FileTree = {
        "solution.py": (
            f"# generated case {seed}\ndef Model(x):\n    return x\n".encode(),
            bool(rng.getrandbits(1)),
        )
    }
    for index in range(rng.randint(6, 13)):
        depth = rng.randint(1, 5)
        directories = [f"level{level}_{rng.randrange(5)}" for level in range(depth)]
        name = f"file_{index}_{rng.randrange(10_000)}.{rng.choice(['cu', 'py', 'dat'])}"
        data = rng.randbytes(rng.randrange(0, 384))
        tree["/".join((*directories, name))] = (data, bool(rng.getrandbits(1)))
    return tree


def _write_tree(root: Path, tree: FileTree, order: list[str] | None = None) -> None:
    root.mkdir(parents=True)
    for relative in order or list(tree):
        data, executable = tree[relative]
        path = root.joinpath(*relative.split("/"))
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        path.chmod(0o751 if executable else 0o640)


def _snapshot(root: Path) -> dict[str, tuple[bytes, int]]:
    return {
        path.relative_to(root).as_posix(): (
            path.read_bytes(),
            stat.S_IMODE(path.stat().st_mode),
        )
        for path in root.rglob("*")
        if path.is_file()
    }


def _expected(
    tree: FileTree, *, projected: bool = False
) -> dict[str, tuple[bytes, int]]:
    result = {}
    for relative, (data, executable) in tree.items():
        destination = relative
        if projected and relative != "solution.py":
            destination = f"scratch/{relative}"
        result[destination] = (data, 0o755 if executable else 0o644)
    return result


def _capture_case(root: Path, tree: FileTree | None = None) -> tuple[Path, str]:
    source = root / "source"
    _write_tree(
        source,
        tree
        or {
            "solution.py": (b"sources = ['kernels/kernel.cu']\n", False),
            "kernels/kernel.cu": (b"// kernel\n", False),
        },
    )
    bundle = root / "bundle"
    return bundle, bundles.capture(source, bundle)


def _manifest(bundle: Path) -> dict[str, object]:
    return json.loads((bundle / "manifest.json").read_bytes())


def _write_manifest(bundle: Path, manifest: object) -> None:
    encoded = (
        json.dumps(
            manifest,
            ensure_ascii=True,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("ascii")
    (bundle / "manifest.json").write_bytes(encoded)


def _bind_run(run: Path, digest: str) -> None:
    (run / "result.json").write_text(json.dumps({"submission_manifest_sha256": digest}))


@pytest.mark.parametrize("seed", ROUND_TRIP_SEEDS, ids=lambda seed: f"seed-{seed:x}")
def test_generated_trees_round_trip_through_every_operation(
    tmp_path: Path, seed: int
) -> None:
    tree = _generated_tree(seed)
    source = tmp_path / "source"
    bundle = tmp_path / "bundle"
    extracted = tmp_path / "extracted"
    recaptured = tmp_path / "recaptured"
    run = tmp_path / "run"
    _write_tree(source, tree)

    digest = bundles.capture(source, bundle)
    assert bundles.verify(bundle, digest) == digest
    loaded_digest, loaded = bundles.load(bundle, digest)
    assert loaded_digest == digest
    assert loaded == {path: data for path, (data, _) in tree.items()}
    assert bundles.extract(bundle, extracted, digest) == digest
    assert _snapshot(extracted) == _expected(tree)
    assert bundles.capture(extracted, recaptured) == digest

    run.mkdir()
    assert bundles.project(bundle, run, digest) == digest
    assert _snapshot(run) == _expected(tree, projected=True)


@pytest.mark.parametrize("seed", ORDER_SEEDS, ids=lambda seed: f"seed-{seed:x}")
def test_creation_order_never_changes_manifest_identity(
    tmp_path: Path, seed: int
) -> None:
    tree = _generated_tree(seed)
    first_order = list(tree)
    second_order = list(tree)
    random.Random(0xD15EA5E + seed).shuffle(second_order)
    _write_tree(tmp_path / "first", tree, first_order)
    _write_tree(tmp_path / "second", tree, second_order)

    first_digest = bundles.capture(tmp_path / "first", tmp_path / "first-bundle")
    second_digest = bundles.capture(tmp_path / "second", tmp_path / "second-bundle")
    assert first_digest == second_digest
    assert (tmp_path / "first-bundle/manifest.json").read_bytes() == (
        tmp_path / "second-bundle/manifest.json"
    ).read_bytes()


def test_projection_keeps_trusted_checker_reports_outside_manifest(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    reports = tmp_path / "check-stage"
    run = tmp_path / "run"
    _write_tree(
        source,
        {
            "solution.py": (b"value = 1\n", False),
            "framework.txt": (b"candidate-value\n", False),
        },
    )
    reports.mkdir()
    (reports / "framework.txt").write_text("trusted-check-value\n")
    run.mkdir()
    bundle = run / "submission"
    digest = bundles.capture(source, bundle, bundles.DERIVED_REPORTS)
    assert "framework.txt" not in {item["path"] for item in _manifest(bundle)["files"]}
    bundles.project(bundle, run, digest, reports)
    assert (run / "scratch/framework.txt").read_text() == "trusted-check-value\n"


def test_random_one_byte_payload_tampering_always_fails(tmp_path: Path) -> None:
    bundle, digest = _capture_case(tmp_path, _generated_tree(0xBAD5EED))
    paths = [
        bundle / "files" / entry["path"]
        for entry in _manifest(bundle)["files"]
        if entry["size"]
    ]
    rng = random.Random(0x51DECA2)

    for _ in range(48):
        path = rng.choice(paths)
        original = path.read_bytes()
        changed = bytearray(original)
        offset = rng.randrange(len(changed))
        changed[offset] ^= rng.randrange(1, 256)
        path.write_bytes(changed)
        try:
            with pytest.raises(bundles.BundleError, match="does not match manifest"):
                bundles.verify(bundle)
        finally:
            path.write_bytes(original)
        assert bundles.verify(bundle, digest) == digest


def test_nested_duplicate_basenames_publish_only_exact_references(
    tmp_path: Path,
) -> None:
    rng = random.Random(0x51DECAB)
    references: list[str] = []
    tree: FileTree = {}
    expected_markers: list[str] = []
    decoy_markers: list[str] = []
    for index in range(12):
        basename = f"kernel_{index}.cu"
        wanted = f"chosen/{rng.randrange(1000)}/deep/{basename}"
        decoy = f"decoys/{rng.randrange(1000)}/deep/{basename}"
        wanted_marker = f"EXACT_SIDE_CAR_{index}_{rng.randrange(1 << 30)}"
        decoy_marker = f"WRONG_SIDE_CAR_{index}_{rng.randrange(1 << 30)}"
        references.append(wanted)
        expected_markers.append(wanted_marker)
        decoy_markers.append(decoy_marker)
        tree[wanted] = (f"// {wanted_marker}\n".encode(), False)
        tree[decoy] = (f"// {decoy_marker}\n".encode(), False)
    solution = (
        "sources = [\n" + "".join(f"    {path!r},\n" for path in references) + "]\n"
    )
    tree["solution.py"] = (solution.encode(), False)

    run = tmp_path / "run"
    source = tmp_path / "source"
    run.mkdir()
    _write_tree(source, tree)
    digest = bundles.capture(source, run / "submission")
    bundles.project(run / "submission", run, digest)
    _bind_run(run, digest)
    rendered = sidecars.augment((run / "solution.py").read_text(), run)

    for path, marker in zip(references, expected_markers):
        assert f"sidecar: {path} " in rendered
        assert marker in rendered
    for marker in decoy_markers:
        assert marker not in rendered


def test_capture_rejects_symlinks_atomically(tmp_path: Path) -> None:
    source = tmp_path / "source"
    _write_tree(source, {"solution.py": (b"pass\n", False)})
    outside = tmp_path / "outside.cu"
    outside.write_bytes(b"outside")
    (source / "kernel.cu").symlink_to(outside)
    with pytest.raises(bundles.BundleError, match="symbolic links"):
        bundles.capture(source, tmp_path / "bundle")
    assert not (tmp_path / "bundle").exists()


def test_capture_rejects_hardlinks_atomically(tmp_path: Path) -> None:
    source = tmp_path / "source"
    _write_tree(
        source,
        {"solution.py": (b"pass\n", False), "first.cu": (b"kernel", False)},
    )
    os.link(source / "first.cu", source / "second.cu")
    with pytest.raises(bundles.BundleError, match="hard-linked"):
        bundles.capture(source, tmp_path / "bundle")
    assert not (tmp_path / "bundle").exists()


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="FIFO creation is unavailable")
def test_capture_rejects_fifo_atomically(tmp_path: Path) -> None:
    source = tmp_path / "source"
    _write_tree(source, {"solution.py": (b"pass\n", False)})
    try:
        os.mkfifo(source / "kernel.pipe")
    except OSError as exc:
        pytest.skip(f"FIFO creation is unsupported here: {exc}")
    with pytest.raises(bundles.BundleError, match="special files"):
        bundles.capture(source, tmp_path / "bundle")
    assert not (tmp_path / "bundle").exists()


@pytest.mark.parametrize(
    "relative,source",
    [
        ("solution.py", b"import os\nos._exit(0)\n"),
        ("solution.py", b"import posix\ngetattr(posix, '_exit')(0)\n"),
        ("solution.py", b"import os\nos.execv('/bin/true', ['true'])\n"),
        ("solution.py", b"import ctypes\nctypes.CDLL(None).syscall(231, 0)\n"),
        ("solution.py", b"import ctypes\nctypes.CDLL(None).exit(0)\n"),
        ("kernel.cu", b"void stop_now() { _exit(0); }\n"),
        ("kernel.cpp", b"syscall(SYS_exit_group, 0);\n"),
    ],
)
def test_capture_rejects_process_level_early_termination(
    tmp_path: Path, relative: str, source: bytes
) -> None:
    tree: FileTree = {"solution.py": (b"pass\n", False)}
    tree[relative] = (source, False)
    submission = tmp_path / "source"
    _write_tree(submission, tree)

    with pytest.raises(bundles.BundleError, match="early termination"):
        bundles.capture(submission, tmp_path / "bundle")

    assert not (tmp_path / "bundle").exists()


def test_capture_rejects_precompiled_binary_with_unknown_suffix(tmp_path: Path) -> None:
    submission = tmp_path / "source"
    _write_tree(
        submission,
        {
            "solution.py": (b"VALUE = 1\n", False),
            "payload.bin": (b"\x7fELF\x02\x01payload", True),
        },
    )

    with pytest.raises(bundles.BundleError, match="precompiled native binaries"):
        bundles.capture(submission, tmp_path / "bundle")


@pytest.mark.parametrize(
    "relative",
    [
        "solution/__init__.py",
        "reference/__init__.py",
        "reference.cpython-311-x86_64-linux-gnu.so",
        "shapes/__init__.py",
        "sota.py.backup",
        "src/eval/correctness.py",
    ],
)
def test_capture_rejects_trusted_module_shadowing(
    tmp_path: Path, relative: str
) -> None:
    submission = tmp_path / "source"
    _write_tree(
        submission,
        {"solution.py": (b"pass\n", False), relative: (b"shadow\n", False)},
    )

    with pytest.raises(bundles.BundleError, match="shadows (?:trusted module|solution.py)"):
        bundles.capture(
            submission,
            tmp_path / "bundle",
            reserved_modules=("reference", "shapes", "sota", "src"),
        )


@pytest.mark.parametrize("relative", ["cache/kernel.pyc", "__pycache__/kernel.py"])
def test_capture_ignores_python_bytecode(tmp_path: Path, relative: str) -> None:
    source = tmp_path / "source"
    _write_tree(
        source,
        {"solution.py": (b"pass\n", False), relative: (b"bytecode", False)},
    )
    bundle = tmp_path / "bundle"
    bundles.capture(source, bundle)
    assert [entry["path"] for entry in _manifest(bundle)["files"]] == ["solution.py"]


@pytest.mark.parametrize(
    "unsafe",
    [
        "../escape.cu",
        "nested/../../escape.cu",
        "/absolute/escape.cu",
        "C:/escape.cu",
        "nested//escape.cu",
        "nested/./escape.cu",
        "nested\\escape.cu",
    ],
)
def test_manifest_paths_cannot_escape_extraction(tmp_path: Path, unsafe: str) -> None:
    bundle, _ = _capture_case(tmp_path)
    manifest = _manifest(bundle)
    entry = next(item for item in manifest["files"] if item["path"] != "solution.py")
    entry["path"] = unsafe
    manifest["files"].sort(key=lambda item: item["path"])
    _write_manifest(bundle, manifest)

    destination = tmp_path / "extracted"
    with pytest.raises(bundles.BundleError, match="unsafe file path"):
        bundles.extract(bundle, destination)
    assert not destination.exists()
    assert not (tmp_path / "escape.cu").exists()


def test_noncanonical_manifest_is_rejected(tmp_path: Path) -> None:
    bundle, _ = _capture_case(tmp_path)
    path = bundle / "manifest.json"
    path.write_text(json.dumps(_manifest(bundle), indent=2) + "\n")
    with pytest.raises(bundles.BundleError, match="not canonical"):
        bundles.verify(bundle)


def test_expected_manifest_digest_must_match(tmp_path: Path) -> None:
    bundle, digest = _capture_case(tmp_path)
    wrong = ("0" if digest[0] != "0" else "1") + digest[1:]
    with pytest.raises(bundles.BundleError, match="manifest digest mismatch"):
        bundles.verify(bundle, wrong)


def test_missing_definite_bundled_sidecar_fails_publication(tmp_path: Path) -> None:
    solution = "sources = ['nested/missing.cu']\n"
    source = tmp_path / "source"
    run = tmp_path / "run"
    run.mkdir()
    _write_tree(source, {"solution.py": (solution.encode(), False)})
    digest = bundles.capture(source, run / "submission")
    bundles.project(run / "submission", run, digest)
    _bind_run(run, digest)
    with pytest.raises(sidecars.SidecarError, match="absent from manifest"):
        sidecars.augment((run / "solution.py").read_text(), run)


def test_bundled_publication_ignores_mutable_projection(tmp_path: Path) -> None:
    source = tmp_path / "source"
    run = tmp_path / "run"
    run.mkdir()
    _write_tree(source, {"solution.py": (b"ORIGINAL_BUNDLE_BYTES = True\n", False)})
    digest = bundles.capture(source, run / "submission")
    bundles.project(run / "submission", run, digest)
    _bind_run(run, digest)
    (run / "solution.py").write_text("MUTABLE_PROJECTION = True\n")

    rendered = sidecars.augment((run / "solution.py").read_text(), run)
    assert "ORIGINAL_BUNDLE_BYTES" in rendered
    assert "MUTABLE_PROJECTION" not in rendered


def test_publication_rejects_bundle_replaced_after_scoring(tmp_path: Path) -> None:
    run = tmp_path / "run"
    first = tmp_path / "first"
    second = tmp_path / "second"
    run.mkdir()
    _write_tree(first, {"solution.py": (b"scored = True\n", False)})
    _write_tree(second, {"solution.py": (b"replacement = True\n", False)})
    digest = bundles.capture(first, run / "submission")
    _bind_run(run, digest)
    bundles.capture(second, tmp_path / "replacement")
    (run / "submission").rename(run / "original-submission")
    (tmp_path / "replacement").rename(run / "submission")

    with pytest.raises(sidecars.SidecarError, match="manifest digest mismatch"):
        sidecars.augment("ignored = True\n", run)


def test_existing_submission_without_manifest_does_not_fall_back(
    tmp_path: Path,
) -> None:
    run = tmp_path / "run"
    (run / "submission").mkdir(parents=True)
    (run / "solution.py").write_text("legacy = True\n")
    _bind_run(run, "0" * 64)
    with pytest.raises(sidecars.SidecarError, match="manifest.json"):
        sidecars.augment((run / "solution.py").read_text(), run)


def test_recorded_bundle_cannot_be_deleted_to_enable_legacy_fallback(
    tmp_path: Path,
) -> None:
    run = tmp_path / "run"
    run.mkdir()
    (run / "solution.py").write_text("mutable_projection = True\n")
    _bind_run(run, "0" * 64)

    with pytest.raises(sidecars.SidecarError, match="bundle is missing"):
        sidecars.augment((run / "solution.py").read_text(), run)


@pytest.mark.parametrize(
    "result",
    [
        "not json\n",
        json.dumps({"submission_bundle_status": "captured"}),
        json.dumps({"submission_manifest_sha256": None}),
    ],
)
def test_broken_bundle_metadata_cannot_enable_legacy_fallback(
    tmp_path: Path, result: str
) -> None:
    run = tmp_path / "run"
    run.mkdir()
    (run / "solution.py").write_text("mutable_projection = True\n")
    (run / "result.json").write_text(result)

    with pytest.raises(sidecars.SidecarError):
        sidecars.augment((run / "solution.py").read_text(), run)


def test_valid_legacy_run_without_bundle_fields_keeps_legacy_rendering(
    tmp_path: Path,
) -> None:
    run = tmp_path / "run"
    run.mkdir()
    solution = "legacy_projection = True\n"
    (run / "solution.py").write_text(solution)
    (run / "result.json").write_text(json.dumps({"correct": True}))

    assert sidecars.augment(solution, run) == solution


def test_shared_runner_uses_separate_check_and_benchmark_replays() -> None:
    text = (REPO / "scripts/lib/run_harness.sh").read_text()
    check = "if prepare_replay_stage check; then"
    benchmark = "if prepare_replay_stage benchmark; then"
    assert 'local stage_dir="$REPLAY_ROOT/$stage_name"' in text
    assert check in text and benchmark in text
    assert text.index(check) < text.index(benchmark)
    assert '"$CHECK_PROBLEM_DIR" check.py' in text
    assert '"$BENCH_PROBLEM_DIR" benchmark.py' in text
    assert '--reserved-module "${t%.py}"' in text
    assert "--reserved-module src" in text
    assert '/bin/rm -rf -- "$REPLAY_ROOT" "$AGENT_UV_OVERLAY"' in text


def test_shared_runner_clears_environment_and_unshares_user_and_network() -> None:
    text = (REPO / "scripts/lib/run_harness.sh").read_text()
    start = text.index("run_replay_stage()")
    end = text.index('\n}\n\nif [ "$TEMPLATE_MUTATED"', start)
    runtime = text[start:end]
    collapsed = " ".join(runtime.replace("\\\n", " ").split())
    assert "KBH_ISOLATION_NETWORK=off" in collapsed
    assert '"$HOST_AGENT_ISOLATOR" /usr/bin/env -i "${replay_env[@]}"' in collapsed
    assert '"$trusted_entrypoint" "$script"' in collapsed
    assert '"$TRUSTED_PYTHON" -I -S' not in runtime
    assert 'WORKSPACE_ROOT="$stage_root"' in runtime
    assert 'WORKSPACE_TRUSTED_PATHS="$replay_trusted_paths"' in runtime
    assert 'run_gpu_locked_timeout check.py' in runtime

    preflight = text.index("replay-preflight")
    assert preflight < start


def test_host_agents_cannot_mutate_grading_roots_or_uv_cache_links() -> None:
    text = (REPO / "scripts/lib/run_harness.sh").read_text()
    start = text.index("cat > \"$HOST_AGENT_ISOLATOR\"")
    end = text.index("\nEOF\nchmod", start)
    isolator = text[start:end]
    assert (
        'for path in "$home" "$repo" "$python_runtime" "$trusted_tools" "$trusted_uv"'
        in isolator
    )
    assert '/usr/bin/mount -o remount,bind,ro "$path"' in isolator
    assert '"$home" "$repo" "$python_runtime" "$trusted_tools" "$trusted_uv"' in isolator
    assert '/usr/bin/mount --bind "$repo/.venv" "$workspace/.venv"' in isolator
    assert 'printf "%s\\n" "$workspace_trusted"' in isolator
    assert 'for path in "$cargo_home" "$rustup_home" "$cuda_oxide" "$cutile_rust"' in isolator
    assert 'printf "%s\\n" "$trusted_worktrees"' in isolator
    assert '"$template_backup" "$trusted_src" "$wrapper_dir" "$control_dir"' in isolator
    assert '"$replay_root"/*' in isolator
    assert "/usr/bin/mount -t overlay overlay" in isolator
    assert "lowerdir=$uv_cache,upperdir=$uv_overlay/upper" in isolator
    assert 'cd "$agent_cwd"' in isolator
    assert 'exec "$setpriv" --no-new-privs --bounding-set=-all' in isolator
    assert '"${timeout_seconds}s" "$HOST_AGENT_ISOLATOR" "$@"' in text
    assert '"$BUNDLE_PYTHON" -I -S - "$1"' in text
    assert "TRUSTED_PATH_MANIFEST=\"$(trusted_path_manifest)\"" in text
    assert "verify_trusted_path_manifest" in text
    assert 'export PATH="$LOCK_WRAPPER_DIR:$KBH_CUDA_HOME/bin:' in text


def test_untrusted_grading_process_cannot_inherit_gpu_lock_descriptors() -> None:
    text = (REPO / "scripts/lib/run_harness.sh").read_text()
    assert '"$real" "$@" 3>&- 9>&-' in text
    replay = text[text.index("run_replay_stage()") :]
    assert '"$HOST_AGENT_ISOLATOR" /usr/bin/env -i' in replay
    assert "run_gpu_locked_timeout check.py" in replay


def test_host_agent_isolator_enforces_read_only_and_copy_on_write_mounts(
    tmp_path: Path,
) -> None:
    unshare = shutil.which("unshare")
    setpriv = shutil.which("setpriv")
    if unshare is None or setpriv is None:
        pytest.skip("util-linux namespace tools are unavailable")

    text = (REPO / "scripts/lib/run_harness.sh").read_text()
    marker = 'cat > "$HOST_AGENT_ISOLATOR" <<\'EOF\'\n'
    isolator = text.split(marker, 1)[1].split("\nEOF\n", 1)[0]

    repo = tmp_path / "repo"
    run = repo / "run"
    workspace = run / "workspace"
    template_backup = run / "template_files"
    trusted_src = run / "trusted_src"
    wrappers = run / "bin"
    replays = run / "replays"
    lock = repo / "lock"
    lock_file = lock / "gpu.lock"
    lock_log = run / "gpu_lock.log"
    control = run / "harness_control"
    runtime = tmp_path / "python-runtime"
    trusted_tools = tmp_path / "trusted-tools"
    trusted_uv = tmp_path / "trusted-uv"
    cache = tmp_path / "uv-cache"
    overlay = run / "agent_uv_overlay"
    for path in (
        workspace,
        workspace / ".venv",
        repo / ".venv",
        template_backup,
        trusted_src,
        wrappers,
        replays,
        lock,
        control,
        runtime,
        trusted_tools,
        cache,
        overlay / "upper",
        overlay / "work",
        overlay / "merged",
    ):
        path.mkdir(parents=True, exist_ok=True)
    (repo / "trusted.txt").write_text("trusted\n")
    (cache / "package.py").write_text("lower\n")
    lock_file.touch()
    lock_log.touch()
    trusted_uv.touch(mode=0o755)

    command = """
if /usr/bin/touch "$1/forbidden" 2>/dev/null; then exit 21; fi
if /usr/bin/touch "$2/forbidden" 2>/dev/null; then exit 22; fi
if /usr/bin/touch "$5/forbidden" 2>/dev/null; then exit 23; fi
if /usr/bin/touch "$6/forbidden" 2>/dev/null; then exit 24; fi
if printf poisoned > "$7" 2>/dev/null; then exit 25; fi
printf 'upper\\n' > "$3/package.py"
grep -qx upper "$3/package.py"
printf 'workspace\\n' > "$4/wrote"
"""
    env = os.environ | {
        "REPO_ROOT": str(repo),
        "RUN_DIR": str(run),
        "KBH_GPU_LOCK_DIR": str(lock),
        "KBH_GPU_LOCK": str(lock_file),
        "KBH_GPU_LOCK_LOG": str(lock_log),
        "TEMPLATE_BACKUP_DIR": str(template_backup),
        "TRUSTED_SRC_BACKUP_DIR": str(trusted_src),
        "LOCK_WRAPPER_DIR": str(wrappers),
        "REPLAY_ROOT": str(replays),
        "HARNESS_CONTROL_DIR": str(control),
        "TRUSTED_PYTHON_RUNTIME": str(runtime),
        "TRUSTED_TOOLS_DIR": str(trusted_tools),
        "REAL_UV": str(trusted_uv),
        "TRUSTED_WORKTREE_ROOTS": str(repo),
        "REAL_UNSHARE": unshare,
        "REAL_SETPRIV": setpriv,
        "UV_CACHE_HOST": str(cache),
        "AGENT_UV_OVERLAY": str(overlay),
        "WORKSPACE_ROOT": str(workspace),
        "WORKSPACE_TRUSTED_PATHS": str(workspace / ".venv"),
        "KBH_CARGO_HOME": str(tmp_path / "cargo"),
        "KBH_RUSTUP_HOME": str(tmp_path / "rustup"),
        "KBH_CUDA_OXIDE_ROOT": str(tmp_path / "cuda-oxide"),
        "KBH_CUTILE_RUST_ROOT": str(tmp_path / "cutile-rust"),
        "DIALECT_CUDA_TOOLKIT": str(runtime),
    }
    try:
        completed = subprocess.run(
            [
                "bash",
                "-s",
                "--",
                "/bin/sh",
                "-u",
                "-c",
                command,
                "probe",
                str(repo),
                str(template_backup),
                str(cache),
                str(workspace),
                str(runtime),
                str(trusted_tools),
                str(trusted_uv),
            ],
            input=isolator,
            text=True,
            capture_output=True,
            cwd=workspace,
            env=env,
            timeout=10,
        )
        if completed.returncode != 0 and "Operation not permitted" in completed.stderr:
            pytest.skip("unprivileged user namespaces are unavailable")
        assert completed.returncode == 0, completed.stderr
        assert (cache / "package.py").read_text() == "lower\n"
        assert (workspace / "wrote").read_text() == "workspace\n"
        assert not (repo / "forbidden").exists()
        assert not (template_backup / "forbidden").exists()
        assert not (runtime / "forbidden").exists()
        assert not (trusted_tools / "forbidden").exists()
        assert trusted_uv.read_bytes() == b""
    finally:
        subprocess.run(
            ["chmod", "-R", "u+rwX", str(tmp_path)],
            check=False,
            capture_output=True,
        )


def test_shared_runner_projects_bundle_before_publishing_result() -> None:
    text = (REPO / "scripts/lib/run_harness.sh").read_text()
    projection = text.index("run_submission_bundle project")
    result = text.index('RESULT_TMP="$(mktemp -p "$RUN_DIR" .result.json.XXXXXX)"')
    publication = text.index('mv -f -- "$RESULT_TMP" "$RUN_DIR/result.json"')
    assert projection < result < publication
    assert '--expect "$SUBMISSION_DIGEST"' in text[projection:result]


@pytest.mark.parametrize("bench", ["hard", "cuda", "mega", "mini"])
def test_sequential_regraders_verify_valid_bundle_binding_and_reject_tampering(
    tmp_path: Path, bench: str
) -> None:
    run = tmp_path / "run"
    run.mkdir()
    source = tmp_path / "source"
    _write_tree(source, {"solution.py": (b"BUNDLE_ONLY = True\n", False)})
    digest = bundles.capture(source, run / "submission")
    (run / "result.json").write_text(
        json.dumps(
            {
                "problem": "probe",
                "submission_bundle_status": "captured",
                "submission_manifest_sha256": digest,
            }
        )
    )
    (run / "repo/problems/probe").mkdir(parents=True)
    env = os.environ | {"KBH_REGRADE_DRY_RUN": "1"}

    completed = subprocess.run(
        [str(REPO / f"benchmarks/{bench}/scripts/regrade_sequential.sh"), str(run)],
        text=True,
        capture_output=True,
        check=False,
        env=env,
    )
    assert completed.returncode == 0, completed.stderr
    assert "[dry-run] would grade" in completed.stdout

    result_path = run / "result.json"
    result = json.loads(result_path.read_text())
    result["submission_manifest_sha256"] = "0" * 64
    result_path.write_text(json.dumps(result))
    completed = subprocess.run(
        [str(REPO / f"benchmarks/{bench}/scripts/regrade_sequential.sh"), str(run)],
        text=True,
        capture_output=True,
        check=False,
        env=env,
    )
    assert completed.returncode == 3
    assert "submission bundle failed verification" in completed.stderr

    result["submission_manifest_sha256"] = digest
    result_path.write_text(json.dumps(result))
    (run / "submission/files/solution.py").write_text("TAMPERED = True\n")
    completed = subprocess.run(
        [str(REPO / f"benchmarks/{bench}/scripts/regrade_sequential.sh"), str(run)],
        text=True,
        capture_output=True,
        check=False,
        env=env,
    )
    assert completed.returncode == 3
    assert "submission bundle failed verification" in completed.stderr


@pytest.mark.parametrize("bench", ["hard", "cuda", "mega", "mini"])
@pytest.mark.parametrize(
    "metadata",
    [
        {"submission_bundle_status": "captured"},
        {"submission_manifest_sha256": "0" * 64},
        {
            "submission_bundle_status": "rejected",
            "submission_manifest_sha256": None,
        },
        {
            "submission_bundle_status": "captured",
            "submission_manifest_sha256": "A" * 64,
        },
    ],
    ids=["missing-digest", "missing-status", "rejected", "uppercase-digest"],
)
def test_sequential_regraders_fail_closed_for_invalid_bundle_metadata(
    tmp_path: Path, bench: str, metadata: dict[str, object]
) -> None:
    run = tmp_path / "run"
    run.mkdir()
    (run / "result.json").write_text(json.dumps(metadata))

    completed = subprocess.run(
        [str(REPO / f"benchmarks/{bench}/scripts/regrade_sequential.sh"), str(run)],
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 3
    assert "invalid or incomplete submission bundle metadata" in completed.stderr


@pytest.mark.parametrize("bench", ["hard", "cuda", "mega", "mini"])
def test_sequential_regraders_restore_bundle_into_independent_stage_roots(
    tmp_path: Path, bench: str
) -> None:
    run = tmp_path / "run"
    workspace = run / "repo"
    problem = workspace / "problems/probe"
    problem.mkdir(parents=True)
    (workspace / "src").mkdir()
    (workspace / "src/trusted.py").write_text("TRUSTED = True\n")
    (workspace / "pyproject.toml").write_text("[project]\nname='probe'\nversion='0'\n")
    (workspace / "uv.lock").write_text("version = 1\n")
    (workspace / ".python-version").write_text("3.11\n")
    for name in (
        "reference.py",
        "sota.py",
        "shapes.py",
        "problem.yaml",
        "check.py",
        "benchmark.py",
        "PROMPT.txt",
    ):
        (problem / name).write_text(f"trusted {name}\n")

    source = tmp_path / "source"
    _write_tree(
        source,
        {
            "solution.py": (b"BUNDLE_SOLUTION = True\n", False),
            "helpers/kernel.cu": (b"// BUNDLE_SIDECAR\n", False),
        },
    )
    digest = bundles.capture(source, run / "submission")
    (run / "result.json").write_text(
        json.dumps(
            {
                "problem": "probe",
                "peak_fraction": 0.25,
                "submission_bundle_status": "captured",
                "submission_manifest_sha256": digest,
            }
        )
    )

    # These mutable compatibility projections must not be consulted for a
    # bundle-bound run.
    (run / "solution.py").write_text("MUTABLE_PROJECTION = True\n")
    (run / "scratch/helpers").mkdir(parents=True)
    (run / "scratch/helpers/kernel.cu").write_text("// MUTABLE_SIDECAR\n")
    (problem / "solution.py").write_text("STALE_WORKSPACE = True\n")

    observed = tmp_path / "observed-stages"
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_uv = fake_bin / "uv"
    fake_uv.write_text(
        """#!/usr/bin/env bash
set -eu
stage="${!#}"
grep -q 'BUNDLE_SOLUTION = True' solution.py
grep -q 'BUNDLE_SIDECAR' helpers/kernel.cu
! grep -q 'MUTABLE_' solution.py helpers/kernel.cu
printf '%s\n' "$PWD" >> "$OBSERVED_STAGES"
case "$stage" in
  check.py)
    printf 'CHECK_MUTATION = True\n' > solution.py
    printf '// CHECK_MUTATION\n' > helpers/kernel.cu
    : > check-only-artifact.py
    printf 'PASS\n'
    ;;
  benchmark.py)
    test ! -e check-only-artifact.py
    printf 'peak_fraction: 0.75\n'
    ;;
  *) exit 90 ;;
esac
"""
    )
    fake_uv.chmod(0o755)
    env = os.environ | {
        "KBH_KEEP_RUN_VENV": "1",
        "KBH_REGRADE_ALLOW_BUSY": "1",
        "OBSERVED_STAGES": str(observed),
        "PATH": f"{fake_bin}:{os.environ['PATH']}",
    }

    completed = subprocess.run(
        [str(REPO / f"benchmarks/{bench}/scripts/regrade_sequential.sh"), str(run)],
        text=True,
        capture_output=True,
        check=False,
        env=env,
    )

    assert completed.returncode == 0, completed.stderr
    stages = observed.read_text().splitlines()
    assert len(stages) == 2
    assert stages[0] != stages[1]
    assert "/check/repo/problems/probe" in stages[0]
    assert "/benchmark/repo/problems/probe" in stages[1]
    result = json.loads((run / "result.json").read_text())
    assert result["correct"] is True
    assert result["peak_fraction"] == 0.75
    assert "MUTABLE_PROJECTION" in (run / "solution.py").read_text()
    assert "MUTABLE_SIDECAR" in (run / "scratch/helpers/kernel.cu").read_text()
    assert not list(run.glob(".regrade-bundle.*"))


@pytest.mark.parametrize("bench", ["hard", "cuda", "mega", "mini"])
def test_sequential_regraders_isolate_python_before_trusted_entrypoint(bench: str) -> None:
    script = (REPO / f"benchmarks/{bench}/scripts/regrade_sequential.sh").read_text()

    assert 'uv run python -I "$TRUSTED_ENTRYPOINT" check.py' in script
    assert 'uv run python -I "$TRUSTED_ENTRYPOINT" benchmark.py' in script


@pytest.mark.parametrize("result", ["not json\n", "[]\n"], ids=["malformed", "non-object"])
@pytest.mark.parametrize("bench", ["hard", "cuda", "mega", "mini"])
def test_legacy_regraders_fail_closed_for_broken_result_metadata(
    tmp_path: Path, bench: str, result: str
) -> None:
    run = tmp_path / "run"
    run.mkdir()
    (run / "solution.py").write_text("pass\n")
    (run / "result.json").write_text(result)

    completed = subprocess.run(
        [str(REPO / f"benchmarks/{bench}/scripts/regrade_sequential.sh"), str(run)],
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 3
    assert "unreadable result metadata" in completed.stderr


@pytest.mark.parametrize("worker", ["brev_worker.sh", "lambda_worker.sh"])
@pytest.mark.parametrize(
    "result",
    [
        json.dumps(
            {
                "submission_bundle_status": "rejected",
                "submission_manifest_sha256": None,
            }
        ),
        "not json\n",
        "[]\n",
        None,
    ],
    ids=["bundle-era", "malformed", "non-object", "missing"],
)
def test_remote_regraders_require_valid_legacy_result_metadata(
    tmp_path: Path, worker: str, result: str | None
) -> None:
    runs = tmp_path / "runs"
    run = runs / "run-id"
    run.mkdir(parents=True)
    (run / "solution.py").write_text("pass\n")
    if result is not None:
        (run / "result.json").write_text(result)
    env = os.environ | {"LAMBDA_API_KEY": "test-only"}

    completed = subprocess.run(
        [str(REPO / f"scripts/{worker}"), "regrade", "worker", "run-id", str(runs)],
        text=True,
        capture_output=True,
        check=False,
        env=env,
    )
    assert completed.returncode == 3
    assert "regrade" in completed.stderr


def _fake_remote_tools(tmp_path: Path) -> tuple[Path, Path]:
    fake_bin = tmp_path / "fake-bin"
    fake_bin.mkdir()
    call_log = tmp_path / "remote-calls.log"
    logger = (
        "#!/bin/sh\n"
        "printf '%s' \"${0##*/}\" >> \"$CALL_LOG\"\n"
        "printf '\\t%s' \"$@\" >> \"$CALL_LOG\"\n"
        "printf '\\n' >> \"$CALL_LOG\"\n"
    )
    for command in ("ssh", "rsync"):
        path = fake_bin / command
        path.write_text(logger)
        path.chmod(0o755)
    curl = fake_bin / "curl"
    curl.write_text(
        logger
        + "printf '%s\\n' "
        + "'{\"data\":[{\"name\":\"worker\",\"id\":\"instance-id\","
        + "\"ip\":\"192.0.2.1\",\"status\":\"active\"}]}'\n"
    )
    curl.chmod(0o755)
    return fake_bin, call_log


def _remote_worker_env(tmp_path: Path, fake_bin: Path, call_log: Path) -> dict[str, str]:
    home = tmp_path / "home"
    home.mkdir()
    return os.environ | {
        "CALL_LOG": str(call_log),
        "HOME": str(home),
        "LAMBDA_API_KEY": "test-only",
        "PATH": f"{fake_bin}:{os.environ['PATH']}",
        "XDG_STATE_HOME": str(tmp_path / "state"),
    }


@pytest.mark.parametrize("worker", ["brev_worker.sh", "lambda_worker.sh"])
def test_remote_regraders_transfer_and_replay_captured_bundle(
    tmp_path: Path, worker: str
) -> None:
    runs = tmp_path / "runs"
    run = runs / "run-id"
    source = tmp_path / "candidate"
    _write_tree(
        source,
        {
            "solution.py": (b"VALUE = 'captured'\n", False),
            "kernels/kernel.cu": (b"// captured sidecar\n", False),
        },
    )
    run.mkdir(parents=True)
    digest = bundles.capture(source, run / "submission")
    (run / "solution.py").write_text("VALUE = 'mutable projection'\n")
    (run / "result.json").write_text(
        json.dumps(
            {
                "problem": "01_fp8_gemm",
                "submission_bundle_status": "captured",
                "submission_manifest_sha256": digest,
            }
        )
    )
    fake_bin, call_log = _fake_remote_tools(tmp_path)

    completed = subprocess.run(
        [str(REPO / "scripts" / worker), "regrade", "worker", run.name, str(runs)],
        text=True,
        capture_output=True,
        check=False,
        env=_remote_worker_env(tmp_path, fake_bin, call_log),
        timeout=10,
    )

    assert completed.returncode == 0, completed.stderr
    calls = call_log.read_text()
    assert f"{run}/submission/" in calls
    assert f"{run}/solution.py" not in calls
    setup = calls.index('rm -rf "$RUN"; mkdir -p "$RUN/repo/problems/01_fp8_gemm"')
    result_upload = calls.index(f"{run}/result.json")
    bundle_upload = calls.index(f"{run}/submission/")
    immutable = calls.index("chmod -R a-w")
    regrade = calls.index("./scripts/regrade_sequential.sh")
    result_pull = calls.rindex(f"kb-regrade/{run.name}/result.json")
    assert setup < result_upload < bundle_upload < immutable < regrade < result_pull
    assert 'cp -a "$BENCH/src" "$RUN/repo/src"' in calls
    assert "pyproject.toml uv.lock .python-version" in calls
    assert "KBH_REGRADE_DECK='problems-h100'" in calls
    assert f"{run}/result.regrade.json" in calls


@pytest.mark.parametrize("worker", ["brev_worker.sh", "lambda_worker.sh"])
def test_remote_regraders_preserve_legacy_solution_and_scratch_transfer(
    tmp_path: Path, worker: str
) -> None:
    runs = tmp_path / "runs"
    run = runs / "run-id"
    scratch = run / "scratch"
    scratch.mkdir(parents=True)
    (run / "solution.py").write_text("VALUE = 'legacy'\n")
    (scratch / "kernel.cu").write_text("// legacy sidecar\n")
    (run / "result.json").write_text(json.dumps({"problem": "01_fp8_gemm"}))
    fake_bin, call_log = _fake_remote_tools(tmp_path)

    completed = subprocess.run(
        [str(REPO / "scripts" / worker), "regrade", "worker", run.name, str(runs)],
        text=True,
        capture_output=True,
        check=False,
        env=_remote_worker_env(tmp_path, fake_bin, call_log),
        timeout=10,
    )

    assert completed.returncode == 0, completed.stderr
    calls = call_log.read_text()
    assert f"{run}/solution.py" in calls
    assert f"{run}/scratch/" in calls
    assert f"{run}/submission/" not in calls
    assert "./scripts/regrade_sequential.sh" in calls
    assert f"{run}/result.regrade.json" in calls


@pytest.mark.parametrize("worker", ["brev_worker.sh", "lambda_worker.sh"])
def test_remote_regraders_reject_changed_captured_bundle_before_transport(
    tmp_path: Path, worker: str
) -> None:
    runs = tmp_path / "runs"
    run = runs / "run-id"
    source = tmp_path / "candidate"
    _write_tree(source, {"solution.py": (b"VALUE = 1\n", False)})
    run.mkdir(parents=True)
    digest = bundles.capture(source, run / "submission")
    (run / "result.json").write_text(
        json.dumps(
            {
                "problem": "01_fp8_gemm",
                "submission_bundle_status": "captured",
                "submission_manifest_sha256": digest,
            }
        )
    )
    (run / "submission/files/solution.py").write_bytes(b"VALUE = 2\n")
    fake_bin, call_log = _fake_remote_tools(tmp_path)

    completed = subprocess.run(
        [str(REPO / "scripts" / worker), "regrade", "worker", run.name, str(runs)],
        text=True,
        capture_output=True,
        check=False,
        env=_remote_worker_env(tmp_path, fake_bin, call_log),
        timeout=10,
    )

    assert completed.returncode == 3
    assert "invalid captured submission bundle" in completed.stderr
    assert not call_log.exists()


@pytest.mark.parametrize("worker", ["brev_worker.sh", "lambda_worker.sh"])
@pytest.mark.parametrize(
    ("status", "digest"),
    [
        ("captured", None),
        ("captured", "A" * 64),
        ("captured", "0" * 63),
        ("rejected", "0" * 64),
        (None, "0" * 64),
    ],
    ids=["missing-digest", "uppercase-digest", "short-digest", "rejected", "missing-status"],
)
def test_remote_regraders_reject_invalid_bundle_binding_before_transport(
    tmp_path: Path, worker: str, status: str | None, digest: str | None
) -> None:
    runs = tmp_path / "runs"
    run = runs / "run-id"
    run.mkdir(parents=True)
    result = {
        "problem": "01_fp8_gemm",
        "submission_bundle_status": status,
        "submission_manifest_sha256": digest,
    }
    (run / "result.json").write_text(json.dumps(result))
    fake_bin, call_log = _fake_remote_tools(tmp_path)

    completed = subprocess.run(
        [str(REPO / "scripts" / worker), "regrade", "worker", run.name, str(runs)],
        text=True,
        capture_output=True,
        check=False,
        env=_remote_worker_env(tmp_path, fake_bin, call_log),
        timeout=10,
    )

    assert completed.returncode == 3
    assert "invalid captured submission bundle metadata" in completed.stderr
    assert not call_log.exists()


@pytest.mark.parametrize("worker", ["brev_worker.sh", "lambda_worker.sh"])
def test_remote_regraders_reject_linked_result_metadata_before_transport(
    tmp_path: Path, worker: str
) -> None:
    runs = tmp_path / "runs"
    run = runs / "run-id"
    run.mkdir(parents=True)
    metadata = tmp_path / "result.json"
    metadata.write_text(json.dumps({"problem": "01_fp8_gemm"}))
    (run / "result.json").symlink_to(metadata)
    (run / "solution.py").write_text("VALUE = 1\n")
    fake_bin, call_log = _fake_remote_tools(tmp_path)

    completed = subprocess.run(
        [str(REPO / "scripts" / worker), "regrade", "worker", run.name, str(runs)],
        text=True,
        capture_output=True,
        check=False,
        env=_remote_worker_env(tmp_path, fake_bin, call_log),
        timeout=10,
    )

    assert completed.returncode == 3
    assert "missing or unreadable result metadata" in completed.stderr
    assert not call_log.exists()


@pytest.mark.parametrize("bench", ["hard", "cuda", "mini"])
def test_shared_bench_publishers_render_verified_bundles(bench: str) -> None:
    publisher = REPO / f"benchmarks/{bench}/scripts/publish_v2.sh"
    text = publisher.read_text()
    assert "from kernel_sidecars import augment" in text
    assert "txt = augment(" in text
