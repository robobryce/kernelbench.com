#!/usr/bin/env bash
# Sequential isolated re-benchmark (standing rule, 2026-07-19).
#
# When a wave ran with many agents concurrent, in-run harness timings are
# time-contaminated even with the path-wrapper GPU lock (per-bench lock dirs,
# absolute-path bypasses, overlapping compile/check). Published peak_fraction /
# ms must come from a re-grade where this process is the ONLY GPU owner.
#
# This replays the exact graded path run_hard.sh uses -- check.py then
# benchmark.py, from the run's own archive workspace, with the run's own
# isolated caches -- one run at a time, refusing to start while another CUDA
# process holds the GPU.
#
# Usage:
#   scripts/regrade_sequential.sh outputs/runs/<run_id> [<run_id> ...]
#   scripts/regrade_sequential.sh outputs/runs/*or-opus*/
#
# Env:
#   KBH_REGRADE_GPU=0            GPU index to grade on (default 0)
#   KBH_REGRADE_ALLOW_BUSY=1     skip the idle-GPU precondition (debug only)
#   KBH_REGRADE_DRY_RUN=1        show what would run, touch nothing
#
# Writes into each result.json, preserving the contended originals:
#   peak_fraction / correct / check_* / benchmark_*   <- clean values
#   regrade: {at, host, gpu, contended: {...}}        <- provenance + originals
#
# check.log / benchmark.log are REPLACED with the clean run and the originals
# moved to *.contended.log. That matters beyond tidiness: the cuda headline
# (per-shape ms -> geomean speedup) is computed downstream from benchmark.log,
# not from result.json, so leaving the contended log in place would publish
# contended milliseconds even with a clean peak_fraction.

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT" || exit 1
TRUSTED_ENTRYPOINT="$REPO_ROOT/src/eval/trusted_entrypoint.py"
if [ ! -f "$TRUSTED_ENTRYPOINT" ]; then
    echo "FATAL: trusted grading entrypoint is missing" >&2
    exit 3
fi
SUBMISSION_BUNDLE_TOOL="$(cd "$SCRIPT_DIR/../../.." && pwd)/scripts/lib/submission_bundle.py"
if [ ! -f "$SUBMISSION_BUNDLE_TOOL" ]; then
    # Thin worker payloads carry shared helpers beside the benchmark scripts.
    SUBMISSION_BUNDLE_TOOL="$SCRIPT_DIR/lib/submission_bundle.py"
fi
SUBMISSION_BUNDLE_SOURCE=""

GPU="${KBH_REGRADE_GPU:-0}"
DRY="${KBH_REGRADE_DRY_RUN:-0}"
CHECK_TIMEOUT="${KBH_CHECK_TIMEOUT_SECONDS:-1800}"

KBH_CUDA_HOME="${KBH_CUDA_HOME:-/usr/local/cuda-13}"
if [ -d "$KBH_CUDA_HOME" ]; then
    export CUDA_HOME="$KBH_CUDA_HOME"
    export PATH="$CUDA_HOME/bin:$PATH"
fi

# The whole point is that we own the GPU, so bypass the lock wrapper rather
# than queue behind it.
export KBH_GPU_LOCK_HELD=1
export CUDA_VISIBLE_DEVICES="$GPU"

if [ "$#" -eq 0 ]; then
    echo "usage: $0 <run_dir> [<run_dir> ...]" >&2
    exit 2
fi

# Refuse to grade while anything else is computing on this GPU. A re-grade that
# races another job is exactly the contamination we are here to remove.
require_idle_gpu() {
    [ "${KBH_REGRADE_ALLOW_BUSY:-0}" = "1" ] && return 0
    local waited=0
    while true; do
        local busy
        local smi_out
        smi_out=$(nvidia-smi -i "$GPU" --query-compute-apps=pid --format=csv,noheader 2>/dev/null)
        busy=$(printf '%s' "$smi_out" | grep -c .)
        busy=${busy:-0}
        if [ "$busy" -eq 0 ]; then
            [ "$waited" -gt 0 ] && echo "    GPU $GPU idle after ${waited}s"
            return 0
        fi
        if [ "$waited" -eq 0 ]; then
            echo "    waiting for GPU $GPU to go idle ($busy compute app(s))..."
        fi
        sleep 30
        waited=$((waited + 30))
        if [ "$waited" -ge 3600 ]; then
            echo "    GPU $GPU still busy after 1h; skipping" >&2
            return 1
        fi
    done
}

purge_untrusted_bytecode() {
    /usr/bin/find "$1" -type d -name __pycache__ -prune \
        -exec /bin/rm -rf -- {} + || return 1
    /usr/bin/find "$1" -type f \( -name '*.pyc' -o -name '*.pyo' \) \
        -delete || return 1
}

run_submission_bundle() {
    # Execute the pre-grading snapshot, so check.py cannot replace the helper
    # that verifies and extracts the independent benchmark replay.
    python3 -I -S -c "$SUBMISSION_BUNDLE_SOURCE" "$@"
}

prepare_bundle_stage() {
    local stage_name="$1"
    local stage_root="$BUNDLE_REPLAY_ROOT/$stage_name/repo"
    local stage_problem="$stage_root/problems/$PROBLEM"
    local item

    if [ ! -d "$WORKSPACE_ROOT/src" ]; then
        echo "FATAL: bundle regrade workspace is missing trusted src/" >&2
        return 1
    fi
    mkdir -p "$stage_root/problems" || return 1
    /bin/cp -a "$WORKSPACE_ROOT/src" "$stage_root/src" || return 1
    for item in pyproject.toml uv.lock .python-version; do
        if [ ! -f "$WORKSPACE_ROOT/$item" ]; then
            echo "FATAL: bundle regrade workspace is missing $item" >&2
            return 1
        fi
        cp -p "$WORKSPACE_ROOT/$item" "$stage_root/$item" || return 1
    done
    run_submission_bundle extract \
        "$RUN_DIR/submission" "$stage_problem" --expect "$BUNDLE_DIGEST" \
        >/dev/null || return 1
    for item in reference.py sota.py shapes.py problem.yaml check.py benchmark.py PROMPT.txt; do
        if [ ! -f "$PROBLEM_DIR/$item" ]; then
            echo "FATAL: bundle regrade workspace is missing trusted $item" >&2
            return 1
        fi
        cp -p "$PROBLEM_DIR/$item" "$stage_problem/$item" || return 1
    done
    purge_untrusted_bytecode "$stage_root" || return 1
    BUNDLE_STAGE_PROBLEM="$stage_problem"
}

PASS=0; FAIL=0; SKIP=0

for RUN_DIR in "$@"; do
    RUN_DIR="${RUN_DIR%/}"
    RID="$(basename "$RUN_DIR")"

    if [ ! -f "$RUN_DIR/result.json" ]; then
        echo "[skip] $RID: no result.json (run never scored)"; SKIP=$((SKIP+1)); continue
    fi
    RESULT_RC=0
    RESULT_METADATA="$(python3 -I -S - "$RUN_DIR/result.json" <<'PY'
import json
import re
import sys

try:
    with open(sys.argv[1], encoding="utf-8") as handle:
        result = json.load(handle)
except (OSError, UnicodeError, ValueError, RecursionError):
    raise SystemExit(2)
if type(result) is not dict:
    raise SystemExit(2)
keys = ("submission_bundle_status", "submission_manifest_sha256")
if not any(key in result for key in keys):
    print("legacy")
    raise SystemExit
status = result.get("submission_bundle_status")
digest = result.get("submission_manifest_sha256")
if status != "captured" or type(digest) is not str or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
    raise SystemExit(3)
print(f"bundle {digest}")
PY
    )" || RESULT_RC=$?
    if [ "$RESULT_RC" -eq 2 ]; then
        echo "FATAL: $RID has unreadable result metadata; refusing to regrade" >&2
        exit 3
    fi
    if [ "$RESULT_RC" -ne 0 ]; then
        echo "FATAL: $RID has invalid or incomplete submission bundle metadata; refusing to regrade" >&2
        exit 3
    fi
    RESULT_KIND="${RESULT_METADATA%% *}"
    if [ "$RESULT_KIND" = "bundle" ]; then
        BUNDLE_DIGEST="${RESULT_METADATA#* }"
        if [ ! -f "$SUBMISSION_BUNDLE_TOOL" ]; then
            echo "FATAL: submission bundle helper is missing; refusing to regrade $RID" >&2
            exit 3
        fi
        SUBMISSION_BUNDLE_SOURCE="$(<"$SUBMISSION_BUNDLE_TOOL")"
        if ! run_submission_bundle verify \
            "$RUN_DIR/submission" --expect "$BUNDLE_DIGEST" >/dev/null; then
            echo "FATAL: $RID submission bundle failed verification; refusing to regrade" >&2
            exit 3
        fi
    elif [ ! -f "$RUN_DIR/solution.py" ]; then
        echo "[skip] $RID: no solution.py"; SKIP=$((SKIP+1)); continue
    fi

    PROBLEM=$(python3 -c "import json,sys;print(json.load(open(sys.argv[1]))['problem'])" "$RUN_DIR/result.json")
    WORKSPACE_ROOT="$RUN_DIR/repo"
    PROBLEM_DIR="$WORKSPACE_ROOT/problems/$PROBLEM"
    if [ ! -d "$PROBLEM_DIR" ]; then
        echo "[skip] $RID: archive workspace missing ($PROBLEM_DIR)"; SKIP=$((SKIP+1)); continue
    fi

    echo "=== $RID ($PROBLEM) ==="

    if [ "$DRY" = "1" ]; then
        echo "    [dry-run] would grade in $PROBLEM_DIR"; continue
    fi

    if [ "$RESULT_KIND" = "legacy" ]; then
        # Preserve the historical projection path only for archives that
        # predate bundle metadata entirely.
        cp "$RUN_DIR/solution.py" "$PROBLEM_DIR/solution.py"
        if [ -d "$RUN_DIR/scratch" ]; then
            cp -r "$RUN_DIR/scratch/." "$PROBLEM_DIR/" 2>/dev/null || true
        fi
        # Purge only after restoring every candidate-controlled archived file.
        purge_untrusted_bytecode "$PROBLEM_DIR" || exit 3
    fi

    require_idle_gpu || { SKIP=$((SKIP+1)); continue; }

    # Same isolated caches the original run used, so a compiled extension
    # resolves the way it did in-session.
    export TORCH_EXTENSIONS_DIR="$RUN_DIR/cache/torch_extensions"
    export TRITON_CACHE_DIR="$RUN_DIR/cache/triton"
    export CUDA_CACHE_PATH="$RUN_DIR/cache/cuda"
    # Fresh bundle stages still use the archived run's one locked environment.
    export UV_PROJECT="$WORKSPACE_ROOT"
    export TMPDIR="$RUN_DIR/tmp" TEMP="$RUN_DIR/tmp" TMP="$RUN_DIR/tmp"
    mkdir -p "$TORCH_EXTENSIONS_DIR" "$TRITON_CACHE_DIR" "$CUDA_CACHE_PATH" "$TMPDIR"

    BENCH_TIMEOUT="${KBH_BENCHMARK_TIMEOUT_SECONDS:-1800}"
    [ "$PROBLEM" = "02_kda_cutlass" ] && BENCH_TIMEOUT="${KBH_BENCHMARK_TIMEOUT_SECONDS:-7200}"

    # Park the contended logs once, then write the clean run to the canonical
    # names so every downstream consumer (ms/speedup extraction, viewers) reads
    # single-owner data. Guarded so a second re-grade cannot clobber the true
    # in-session original.
    for L in check benchmark; do
        if [ -f "$RUN_DIR/$L.log" ] && [ ! -f "$RUN_DIR/$L.contended.log" ]; then
            mv "$RUN_DIR/$L.log" "$RUN_DIR/$L.contended.log"
        fi
    done
    CLOG="$RUN_DIR/check.log"
    BLOG="$RUN_DIR/benchmark.log"

    CHECK_PROBLEM_DIR="$PROBLEM_DIR"
    BUNDLE_REPLAY_ROOT=""
    if [ "$RESULT_KIND" = "bundle" ]; then
        BUNDLE_REPLAY_ROOT="$(mktemp -d "$RUN_DIR/.regrade-bundle.XXXXXX")" || exit 3
        if ! prepare_bundle_stage check; then
            /bin/rm -rf -- "$BUNDLE_REPLAY_ROOT"
            echo "FATAL: could not restore verified bundle for $RID check" >&2
            exit 3
        fi
        CHECK_PROBLEM_DIR="$BUNDLE_STAGE_PROBLEM"
    fi

    echo "    check.py..."
    C0=$(date +%s); CEXIT=0
    (cd "$CHECK_PROBLEM_DIR" && timeout "$CHECK_TIMEOUT" \
        uv run python -I "$TRUSTED_ENTRYPOINT" check.py) > "$CLOG" 2>&1 || CEXIT=$?
    CEL=$(( $(date +%s) - C0 ))

    CORRECT=false; SCORE=null; BEXIT=null; BEL=null
    CPASS_COUNT=$(grep -axc 'PASS' "$CLOG" || true)
    if [ "$CEXIT" -eq 0 ] && [ "$CPASS_COUNT" -eq 1 ]; then
        CORRECT=true
        echo "    benchmark.py..."
        BENCH_PROBLEM_DIR="$PROBLEM_DIR"
        if [ "$RESULT_KIND" = "bundle" ]; then
            if ! prepare_bundle_stage benchmark; then
                /bin/rm -rf -- "$BUNDLE_REPLAY_ROOT"
                echo "FATAL: could not restore verified bundle for $RID benchmark" >&2
                exit 3
            fi
            BENCH_PROBLEM_DIR="$BUNDLE_STAGE_PROBLEM"
        else
            purge_untrusted_bytecode "$PROBLEM_DIR" || exit 3
        fi
        B0=$(date +%s); BEXIT=0
        (cd "$BENCH_PROBLEM_DIR" && timeout "$BENCH_TIMEOUT" \
            uv run python -I "$TRUSTED_ENTRYPOINT" benchmark.py) > "$BLOG" 2>&1 || BEXIT=$?
        BEL=$(( $(date +%s) - B0 ))
        SCORE_RE='^peak_fraction:[[:space:]]*([0-9]+([.][0-9]*)?|[.][0-9]+)([eE][+-]?[0-9]+)?[[:space:]]*$'
        SCORE_COUNT=$(grep -aEc "$SCORE_RE" "$BLOG" || true)
        if [ "$BEXIT" -eq 0 ] && [ "$SCORE_COUNT" -eq 1 ]; then
            SCORE=$(grep -aE "$SCORE_RE" "$BLOG" \
                | sed -E 's/^peak_fraction:[[:space:]]*//; s/[[:space:]]*$//')
        else
            SCORE=null
            echo "    benchmark FAILED (exit $BEXIT, score markers $SCORE_COUNT) -- see $BLOG"
        fi
    else
        echo "    check FAILED (exit $CEXIT) -- see $CLOG"
    fi

    RID="$RID" CORRECT="$CORRECT" SCORE="$SCORE" CEXIT="$CEXIT" CEL="$CEL" \
    BEXIT="$BEXIT" BEL="$BEL" GPU="$GPU" \
    python3 - "$RUN_DIR/result.json" <<'PY'
import json, os, socket, subprocess, sys

path = sys.argv[1]
with open(path) as f:
    r = json.load(f)

def num(v):
    return None if v in (None, "", "null") else (float(v) if "." in str(v) else int(v))

# Keep the contended originals so an audit can always see what shifted.
contended = {k: r.get(k) for k in (
    "correct", "peak_fraction", "check_exit_code", "benchmark_exit_code",
    "check_elapsed_seconds", "benchmark_elapsed_seconds")}

try:
    gpu_name = subprocess.check_output(
        ["nvidia-smi", "-i", os.environ["GPU"], "--query-gpu=name",
         "--format=csv,noheader"], text=True).strip()
except Exception:
    gpu_name = None

r["correct"] = os.environ["CORRECT"] == "true"
r["peak_fraction"] = num(os.environ["SCORE"])
r["check_exit_code"] = num(os.environ["CEXIT"])
r["benchmark_exit_code"] = num(os.environ["BEXIT"])
r["check_elapsed_seconds"] = num(os.environ["CEL"])
r["benchmark_elapsed_seconds"] = num(os.environ["BEL"])
r["regrade"] = {
    "at": subprocess.check_output(["date", "-Is"], text=True).strip(),
    "host": socket.gethostname(),
    "gpu_index": int(os.environ["GPU"]),
    "gpu_name": gpu_name,
    "mode": "sequential_isolated",
    "contended": contended,
}

with open(path, "w") as f:
    json.dump(r, f, indent=4)

old, new = contended["peak_fraction"], r["peak_fraction"]
delta = ""
if isinstance(old, (int, float)) and isinstance(new, (int, float)) and old:
    delta = "  (%+.1f%%)" % ((new - old) / old * 100)
print("    correct=%s  peak %s -> %s%s" % (r["correct"], old, new, delta))
PY

    if [ -n "$BUNDLE_REPLAY_ROOT" ]; then
        /bin/rm -rf -- "$BUNDLE_REPLAY_ROOT"
    fi

    # uv run recreates repo/.venv during regrade; drop it again so archives stay thin.
    # shellcheck source=../../../scripts/lib/strip_run_venv.sh
    . "$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)/scripts/lib/strip_run_venv.sh"
    strip_run_venv "$RUN_DIR"

    if [ "$CORRECT" = "true" ]; then PASS=$((PASS+1)); else FAIL=$((FAIL+1)); fi
done

echo "========================================"
echo "re-graded: $PASS correct, $FAIL failed, $SKIP skipped"
echo "========================================"
