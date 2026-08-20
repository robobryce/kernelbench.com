#!/usr/bin/env bash
# Brev GPU worker lifecycle for the hard bench, driven from the control plane
# (Mac or anvil). Wraps: provision -> sync -> bootstrap -> run/regrade -> pull
# -> verified teardown. `kb brev ...` shells out here.
#
#   brev_worker.sh up <name> [type]             create instance (default hyperstack_H100) + wait + refresh ssh
#   brev_worker.sh sync <name>                  rsync thin bench (KB_BREV_BENCH, default hard) -> <name>:kb-<bench>/
#   brev_worker.sh bootstrap <name> [--agents]  uv + torch + pinned CUDA dialects; --agents adds agent CLIs + auth
#   brev_worker.sh run <name> <harness> <model> <problem> [effort]   detached agent session (problems root auto)
#   brev_worker.sh regrade <name> <run_id> [runs_dir]   re-grade an archived solution.py: check.py then benchmark.py, sequentially
#   brev_worker.sh pull <name>                  rsync outputs/runs back (thin) into outputs/runs-brev-<name>/
#   brev_worker.sh down <name>                  teardown via brev_teardown.sh, verified against brev ls
#
# Env: KB_BREV_PROBLEMS_ROOT (default problems-h100), KB_BREV_GPU (default H100),
#      KBH_HARDWARE (default H100) for roofline peaks on regrade.
set -euo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"          # repo root
# Bench selected by KB_BREV_BENCH (default hard) — mirrors KB_LAMBDA_BENCH.
BENCH="${KB_BREV_BENCH:-hard}"
BENCH_DIR="$HERE/benchmarks/$BENCH"
REMOTE_DIR="kb-$BENCH"
[ -d "$BENCH_DIR" ] || { echo "ERROR: unknown bench '$BENCH'" >&2; exit 1; }
BREV="${BREV:-brev}"
CMD="${1:?usage: brev_worker.sh <up|sync|bootstrap|run|regrade|pull|down> <name> ...}"
NAME="${2:?instance name required}"
shift 2
S=(ssh -F "$HOME/.brev/ssh_config" -o StrictHostKeyChecking=no)
case "$BENCH" in
  mega) PROBLEMS_ROOT="${KB_BREV_PROBLEMS_ROOT:-problems}" ;;
  *)    PROBLEMS_ROOT="${KB_BREV_PROBLEMS_ROOT:-problems-h100}" ;;
esac

# Keys a worker actually needs; never ship the whole ~/.env_vars.
ENV_ALLOWLIST='KIMI_API_KEY|MOONSHOT_API_KEY|ZAI_API_KEY|MINIMAX_API_KEY|DEEPSEEK_API_KEY|LONGCAT_API_KEY|TENCENT_API_KEY|DASHSCOPE_API_KEY|QWEN_API_KEY|OPENROUTER_API_KEY|OPENAI_API_KEY|GEMINI_API_KEY|ANTHROPIC_API_KEY|CLAUDE_CODE_OAUTH_TOKEN'

ensure_reachable() {
  for _ in 1 2 3; do
    "${S[@]}" -o ConnectTimeout=15 "$NAME" true 2>/dev/null && return 0
    echo "  (host unreachable -> brev refresh)"
    "$BREV" refresh >/dev/null 2>&1 || true
    sleep 3
  done
  echo "ERROR: $NAME unreachable after brev refresh; check 'brev ls'" >&2
  exit 1
}

apply_worker_torch_index() {
  "${S[@]}" "$NAME" "cd ~/$REMOTE_DIR"' && if ! grep -q pytorch-cu128 pyproject.toml; then cat >> pyproject.toml <<TOML

[[tool.uv.index]]
name = "pytorch-cu128"
url = "https://download.pytorch.org/whl/cu128"
explicit = true

[tool.uv.sources]
torch = { index = "pytorch-cu128" }
TOML
fi
rm -f uv.lock
export PATH="$HOME/.local/bin:$PATH"
uv sync'
}

case "$CMD" in
  up)
    # arg = brev instance type (from `brev search`), e.g. hyperstack_H100
    TYPE="${1:-${KB_BREV_TYPE:-hyperstack_H100}}"
    echo "[up] brev create $NAME --type $TYPE"
    "$BREV" create "$NAME" --type "$TYPE"
    echo "[up] waiting for RUNNING/READY ..."
    for _ in $(seq 1 60); do
      row="$("$BREV" ls 2>/dev/null | awk -v n="$NAME" '$1==n')"
      echo "  $row"
      grep -q "RUNNING" <<<"$row" && grep -q "READY" <<<"$row" && break
      sleep 15
    done
    "$BREV" refresh >/dev/null 2>&1 || true
    ensure_reachable
    echo "[up] $NAME reachable"
    ;;

  sync)
    ensure_reachable
    echo "[sync] thin $BENCH bench -> $NAME:$REMOTE_DIR/"
    REMOTE_TORCH_PATCHED=0
    if "${S[@]}" "$NAME" "grep -q pytorch-cu128 $REMOTE_DIR/pyproject.toml" 2>/dev/null; then
      REMOTE_TORCH_PATCHED=1
    fi
    SYNC_EXCLUDES=(--exclude outputs --exclude __pycache__ --exclude '.venv' --exclude '*.pyc'
      --exclude .git --exclude 'docs/refs'
      --exclude 'results/annotations' --exclude 'docs/*case_stud*')
    # Always replace the remote project metadata. The worker patch helper reapplies the
    # node-specific Torch index and regenerates its lock from these current
    # dependencies; preserving a patched remote copy can silently omit new
    # dependencies added by the repository.
    rsync -az -e "${S[*]}" "${SYNC_EXCLUDES[@]}" "$BENCH_DIR/" "$NAME:$REMOTE_DIR/"
    if [ "$REMOTE_TORCH_PATCHED" = 1 ]; then
      echo "[sync] reapplying node torch-index patch to current project metadata"
      apply_worker_torch_index
    fi
    # Single-GPU benches' run_hard.sh wraps the shared runner at
    # <monorepo>/scripts/lib/; ship the lib INTO the bench dir (wrapper falls
    # back to it on thin-synced nodes).
    rsync -az -e "${S[*]}" "$HERE/scripts/lib/" "$NAME:$REMOTE_DIR/scripts/lib/"
    TMPENV="$(mktemp)"
    grep -E "^(export )?($ENV_ALLOWLIST)=" ~/.env_vars > "$TMPENV" || true
    rsync -az -e "${S[*]}" "$TMPENV" "$NAME:.env_vars"
    rm -f "$TMPENV"
    ;;

  bootstrap)
    ensure_reachable
    AGENTS=0; [ "${1:-}" = "--agents" ] && AGENTS=1
    echo "[bootstrap] uv + torch + pinned CUDA dialects (agents=$AGENTS)"
    "${S[@]}" "$NAME" 'command -v uv >/dev/null 2>&1 || curl -LsSf https://astral.sh/uv/install.sh | sh'
    # cu128 torch: stock brev images ship R570-class drivers; the repo cu130
    # pin needs R580. Same override the mega cloud bootstrap uses.
    apply_worker_torch_index
    case "$BENCH" in
      hard|cuda|mini)
        "${S[@]}" "$NAME" "cd ~/$REMOTE_DIR && export PATH=\"\$HOME/.local/bin:\$PATH\" && bash scripts/lib/bootstrap_dialects.sh"
        ;;
    esac
    if [ "$AGENTS" = 1 ]; then
      "${S[@]}" "$NAME" 'command -v node >/dev/null 2>&1 || { curl -fsSL https://deb.nodesource.com/setup_22.x | sudo -E bash - >/dev/null 2>&1 && sudo apt-get install -y nodejs >/dev/null 2>&1; }
        command -v bwrap >/dev/null 2>&1 || sudo apt-get install -y -qq bubblewrap >/dev/null 2>&1
        command -v codex >/dev/null 2>&1 || sudo npm i -g @openai/codex >/dev/null 2>&1
        command -v claude >/dev/null 2>&1 || sudo npm i -g @anthropic-ai/claude-code >/dev/null 2>&1'
      "${S[@]}" "$NAME" 'mkdir -p .codex .claude'
      rsync -az -e "${S[*]}" ~/.codex/auth.json "$NAME:.codex/auth.json" 2>/dev/null || true
      rsync -az -e "${S[*]}" ~/.claude/.credentials.json "$NAME:.claude/.credentials.json" 2>/dev/null || true
    fi
    "${S[@]}" "$NAME" 'export PATH="$HOME/.local/bin:$PATH"; cd ~/'"$REMOTE_DIR"' && uv run python -c "import torch;print(\"torch\",torch.__version__,\"cuda\",torch.cuda.is_available(),torch.cuda.get_device_name(0))"'
    ;;

  run)
    HARNESS="${1:?harness}"; MODEL="${2:?model}"; PROBLEM="${3:?problem}"; EFFORT="${4:-}"
    ensure_reachable
    echo "[run] detached: $HARNESS $MODEL $PROBLEMS_ROOT/$PROBLEM $EFFORT"
    "${S[@]}" "$NAME" "cd ~/$REMOTE_DIR && mkdir -p outputs && setsid nohup env KBH_AGENT_CONTAINER=0 BUDGET_SECONDS=0 ${KB_BREV_RUN_ENV:-} ./scripts/run_hard.sh $HARNESS $MODEL $PROBLEMS_ROOT/$PROBLEM $EFFORT > outputs/kb_run_\$(basename $PROBLEM).log 2>&1 < /dev/null & echo launched PID \$!"
    echo "Poll:  ${S[*]} $NAME 'tail -20 ~/$REMOTE_DIR/outputs/kb_run_*.log'"
    ;;

  regrade)
    RID="${1:?run_id}"; RUNS_DIR="${2:-$BENCH_DIR/outputs/runs-h100}"
    if [ "${#RID}" -gt 255 ] || \
       [[ ! "$RID" =~ ^[A-Za-z0-9][A-Za-z0-9._@%+=,-]*$ ]]; then
      echo "FATAL: unsafe run_id for remote regrade: $RID" >&2
      exit 3
    fi
    SRC="$RUNS_DIR/$RID"
    if RESULT_META="$({ /usr/bin/python3 -I -S - "$SRC/result.json" "$HERE" <<'PY'
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, sys.argv[2])
from scripts.lib.submission_bundle import BundleError, read_regular

try:
    result = json.loads(read_regular(Path(sys.argv[1]), 1024 * 1024))
except (BundleError, OSError, UnicodeError, ValueError, RecursionError):
    raise SystemExit(2)
if type(result) is not dict:
    raise SystemExit(2)
problem = result.get("problem")
if type(problem) is not str or any(character in problem for character in "\t\r\n"):
    raise SystemExit(2)
bundle = any(
    name in result
    for name in ("submission_bundle_status", "submission_manifest_sha256")
)
digest = ""
if bundle:
    digest = result.get("submission_manifest_sha256")
    if (
        result.get("submission_bundle_status") != "captured"
        or type(digest) is not str
        or re.fullmatch(r"[0-9a-f]{64}", digest) is None
    ):
        raise SystemExit(3)
print("\t".join(("bundle" if bundle else "legacy", problem, digest)))
PY
    } 2>/dev/null)"; then
      IFS=$'\t' read -r RESULT_KIND PROBLEM SUBMISSION_DIGEST <<<"$RESULT_META"
    else
      METADATA_STATUS=$?
      if [ "$METADATA_STATUS" -eq 3 ]; then
        echo "FATAL: $RID has invalid captured submission bundle metadata; refusing remote regrade" >&2
      else
        echo "FATAL: $RID has missing or unreadable result metadata; refusing remote regrade" >&2
      fi
      exit 3
    fi
    if [ "${#PROBLEM}" -gt 255 ] || \
       [[ ! "$PROBLEM" =~ ^[0-9]{2}_[a-z0-9]+(_[a-z0-9]+)*$ ]]; then
      echo "FATAL: unsafe problem for remote regrade: $PROBLEM" >&2
      exit 3
    fi
    if [ "$RESULT_KIND" = "bundle" ]; then
      if ! /usr/bin/python3 -I -S "$HERE/scripts/lib/submission_bundle.py" \
          verify "$SRC/submission" --expect "$SUBMISSION_DIGEST" >/dev/null; then
        echo "FATAL: $RID has an invalid captured submission bundle; refusing remote regrade" >&2
        exit 3
      fi
    else
      [ -f "$SRC/solution.py" ] || { echo "no solution.py in $SRC" >&2; exit 1; }
    fi
    ensure_reachable
    echo "[regrade] $RID -> $PROBLEMS_ROOT/$PROBLEM (sequential, no other GPU jobs)"
    REMOTE_RUN="kb-regrade/$RID"
    "${S[@]}" "$NAME" "set -e; \
      RUN=\"\$HOME/$REMOTE_RUN\"; \
      BENCH=\"\$HOME/$REMOTE_DIR\"; \
      TEMPLATE=\"\$BENCH/$PROBLEMS_ROOT/$PROBLEM\"; \
      rm -rf \"\$RUN\"; mkdir -p \"\$RUN/repo/problems/$PROBLEM\"; \
      cp -a \"\$BENCH/src\" \"\$RUN/repo/src\"; \
      for item in pyproject.toml uv.lock .python-version; do cp -p \"\$BENCH/\$item\" \"\$RUN/repo/\$item\"; done; \
      for item in reference.py sota.py shapes.py problem.yaml check.py benchmark.py PROMPT.txt; do \
        cp -p \"\$TEMPLATE/\$item\" \"\$RUN/repo/problems/$PROBLEM/\$item\"; \
      done; \
      if [ -f \"\$TEMPLATE/baseline.py\" ]; then cp -p \"\$TEMPLATE/baseline.py\" \"\$RUN/repo/problems/$PROBLEM/baseline.py\"; fi"
    rsync -az -e "${S[*]}" "$SRC/result.json" "$NAME:$REMOTE_RUN/result.json"
    if [ "$RESULT_KIND" = "bundle" ]; then
      rsync -az --delete -e "${S[*]}" \
        "$SRC/submission/" "$NAME:$REMOTE_RUN/submission/"
      "${S[@]}" "$NAME" "chmod -R a-w \"\$HOME/$REMOTE_RUN/submission\""
    else
      rsync -az -e "${S[*]}" "$SRC/solution.py" "$NAME:$REMOTE_RUN/solution.py"
      if [ -d "$SRC/scratch" ]; then
        rsync -az --delete -e "${S[*]}" "$SRC/scratch/" "$NAME:$REMOTE_RUN/scratch/"
      fi
    fi
    "${S[@]}" "$NAME" "export PATH=\"\$HOME/.local/bin:\$PATH\"; \
      cd \"\$HOME/$REMOTE_DIR\"; \
      env KBH_REGRADE_DECK='$PROBLEMS_ROOT' KBH_HARDWARE=${KBH_HARDWARE:-H100} \
        ./scripts/regrade_sequential.sh \"\$HOME/$REMOTE_RUN\""
    rsync -az -e "${S[*]}" "$NAME:$REMOTE_RUN/result.json" "$SRC/result.regrade.json"
    echo "[regrade] updated metadata -> $SRC/result.regrade.json"
    ;;

  pull)
    ensure_reachable
    DEST="$BENCH_DIR/outputs/runs-brev-$NAME"
    mkdir -p "$DEST"
    echo "[pull] $NAME:$REMOTE_DIR/outputs/runs/ -> $DEST (thin)"
    rsync -az -e "${S[*]}" \
      --exclude '.venv' --exclude 'cache' --exclude 'tmp' --exclude 'container_uv_cache' \
      "$NAME:$REMOTE_DIR/outputs/runs/" "$DEST/"
    ;;

  down)
    exec "$HERE/scripts/brev_teardown.sh" "$NAME"
    ;;

  *)
    echo "unknown subcommand: $CMD" >&2
    exit 2
    ;;
esac
