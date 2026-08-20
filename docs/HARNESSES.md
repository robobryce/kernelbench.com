# KernelBench harnesses

These tables are the single source of truth for runner harness names, transports, and credentials; `AGENTS.md` links here instead of duplicating branch details. Rows come from the primary harness `case` statements in the shared single-GPU runner `scripts/lib/run_harness.sh` (dispatch for hard, cuda, and mini — their `run_hard.sh` files are thin identity wrappers), `benchmarks/mega/scripts/run_hard.sh` (deliberate fork), and `benchmarks/multi/scripts/run_agent.sh`.

“CLI login” means the branch does not enforce one environment key: the named CLI may use its existing login/config or one of the noted optional keys. Single-GPU runners source `~/.env_vars`; Multi provider routes load the worker's `~/.kbm_env` where noted.

## Submission capture and grading

The Hard, CUDA, Mini, and Mega runners capture `solution.py` and its regular
sidecar files immediately after the agent exits. A canonical manifest records
each relative path, mode, size, and SHA-256 digest. Links, special files,
hardlinks, and submissions over the fixed file/size limits are rejected;
generated Python bytecode is excluded. The archive's `solution.py` and
submission sidecars are then regenerated from that bundle rather than copied
from the mutable agent workspace. Checker-generated `framework.txt` and
`cuda_language.json` reports are archived separately under `scratch/`.
Legacy sequential and remote regraders refuse bundle-bound runs: they cannot
replace a score while preserving the original manifest binding. Such a run
must be graded again through the isolated bundle path.

In the shared runner, correctness and performance use separate verified
extractions with fresh compiler caches. Mega likewise restores the verified
candidate before each stage and protects the manifest and trusted entrypoint
from the candidate. Grading uses the canonical benchmark environment, including its
reviewed Python startup files, and do not inherit
API credential environment variables, and run in private network and process
namespaces. There is no externally routed interface, and remaining descendants
are killed when each stage exits. Direct package downloads and remote repository
clones therefore fail while the submission is being graded. The manifest digest
and capture status are stored in `result.json`.

The AAB/Codex process, its agent-visible `check.py`/`benchmark.py` commands, and
the final check/benchmark replay all enter the same mount/process isolation
helper. The source repository, home tree, canonical `.venv`, Python runtime,
trusted helpers, problem templates, `src/`, project metadata, `uv` cache, Rust
toolchain, CUDA Oxide source, and cuTile Rust source are read-only. Only the
submission/sidecar area, compiler caches, logs, and per-run agent state remain
writable. Agent-executed commands and final replay are both offline; final
replay enforces that policy with a private network namespace while Codex uses
its OS sandbox. Both use the same dependency and filesystem contract. The Codex inference
client remains online, but model-generated commands run in Codex's
`workspace-write` sandbox with command network disabled and approvals set to
`never`; a preflight proves DNS lookup, `curl`, and a direct Python socket all
fail before inference starts. The runner fails before launching the agent
unless the immutable environment already provides CUDA C++, CUDA Oxide,
CuTe DSL, Triton, cuTile Python, and cuTile Rust; it never installs a missing
toolchain during a run.

The canonical worker setup commands, `scripts/brev_worker.sh bootstrap` and
`scripts/lambda_worker.sh bootstrap`, provision that environment
deterministically. They install Triton 3.6.0, CUTLASS/CuTe DSL 4.7.0, and
cuTile Python 1.5.0 from the committed Python locks; CUDA Oxide commit
`6c5458fe991bbde32c5bee74d87822aef1b5a691` with Rust
`nightly-2026-04-03`; and cuTile Rust commit
`a3ed99d225befcb19f75ec8d81708eb35818fee2` with Rust 1.89.0 and CUDA Tile
submodule `0859212ad19f71133a9b940c05323286cbf28a05`. The complete CUDA 13.3.1
toolkit lives outside the Python environment in a versioned Cargo-home path.
Bootstrap performs locked Cargo checks, while each run compiles a CUDA source,
checks the pinned revisions offline, validates the Python imports, and runs the
existing GPU preflight before agent inference.

Mega grading additionally uses private mount, PID, and network namespaces with
the home tree read-only and only the candidate workspace and compiler caches
writable. Multi keeps its existing runner and archive format.

Each extraction starts through a trusted entrypoint that requires normal Python
fallthrough; candidate-triggered `SystemExit(0)` and direct process-exit calls
are failures, and source files containing process-level early-termination
primitives are rejected during capture. Correctness
accepts exactly one standalone `PASS`, performance accepts exactly one complete
score line, and both require a zero process exit. The correctness scripts also
run seeded, generated structural cases whose plan is frozen before importing the
submission. These checks close the known early-exit and memorized-input attacks,
but they do not turn same-interpreter Python execution into a complete sandbox.

| Harness | Endpoint/transport | Required env key(s) | Benches that have it | Notes/quirks |
| --- | --- | --- | --- | --- |
| `claude` | Native Claude Code to Anthropic | CLI login; optionally `CLAUDE_CODE_OAUTH_TOKEN`, `ANTHROPIC_API_KEY`, or `ANTHROPIC_AUTH_TOKEN` | hard, cuda, mini, mega, multi | The branch enforces no key. Reasoning effort is forwarded; single-GPU settings disable fast mode and enable thinking. |
| `ccr-claude` | Claude Code to local ccr-rust Anthropic-compatible router at `CCR_BASE_URL` or `http://127.0.0.1:3456` | None in runner; ccr-rust must already have upstream auth | hard, cuda, mini, mega | The model argument is the upstream provider's model ID. |
| `zai-claude` | Claude Code to Z.ai Anthropic API at `https://api.z.ai/api/anthropic` | `ZAI_API_KEY` | hard, cuda, mini, mega, multi | Claude aliases are remapped to the requested model. Multi reads the key from `~/.kbm_env`. |
| `minimax-claude` | Claude Code to `MINIMAX_ANTHROPIC_BASE_URL` or `https://api.minimax.io/anthropic` | `MINIMAX_API_KEY` | hard, cuda, mini, mega | Keeps MiniMax routing separate from native Claude defaults and remaps Claude aliases. |
| `kimi-claude` | Claude Code to `KIMI_ANTHROPIC_BASE_URL` or `https://api.moonshot.ai/anthropic` | `KIMI_API_KEY` | hard, cuda, mini, mega, multi | Kimi's coding route expects thinking mode. Multi reads the key from `~/.kbm_env`. |
| `kinetic-claude` | Claude Code to `KINETIC_ANTHROPIC_BASE_URL` or `https://api.moonshot.ai/anthropic` | `MOONSHOT_API_KEY` | hard, cuda, mini, mega | For `kinetic-0715`; `KIMI_API_KEY` is not interchangeable. Pins max effort, disables tool search, and pins the subagent model. |
| `or-fable` / `openrouter-fable` / `or-opus` / `openrouter-opus` | Claude Code to OpenRouter's Anthropic API at `OR_FABLE_BASE_URL` or `https://openrouter.ai/api` | `OPENROUTER_API_KEY` | hard, cuda, mini (shared runner), mega | Maps bare Fable/Opus aliases to Anthropic slugs and requests one-hour prompt caching. Other OpenRouter slugs pass through unchanged; Qwen 3.8 Max is `qwen/qwen3.8-max` and supports `xhigh` effort. |
| `longcat-claude` | Claude Code to `LONGCAT_ANTHROPIC_BASE_URL` or `https://api.longcat.chat/anthropic` | `LONGCAT_API_KEY` | hard, cuda, mini, mega | Claude aliases map to LongCat-2.0; the route raises the default output limit. |
| `hy3` / `hy3-claude` | OpenCode to Tencent TokenHub's OpenAI-compatible API at `HY3_TOKENHUB_BASE_URL` or `https://tokenhub.tencentmaas.com/v1` | `TENCENT_API_KEY` | hard, cuda, mini, mega | `hy3-claude` is a legacy name, not Claude Code. Only model `hy3` is accepted; preview/OpenRouter slugs are rejected; effort maps to `high` or `no_think`. |
| `tinker` / `inkling` (shared-runner branch) | OpenCode to Tinker's OpenAI-compatible API at `TINKER_BASE_URL` or `https://tinker.thinkingmachines.dev/services/tinker-prod/oai/api/v1` | `THINKING_MACHINES_API_KEY` or `TINKER_API_KEY` | hard, cuda, mini (shared runner) | Both labels serve `thinkingmachines/Inkling` directly and auto-continue the same session when it asks to proceed. |
| `tinker` (Mega branch) | OpenCode to Tinker's OpenAI-compatible API at `TINKER_BASE_URL` or `https://tinker.thinkingmachines.dev/services/tinker-prod/oai/api/v1` | `THINKING_MACHINES_API_KEY` or `TINKER_API_KEY` | mega | Direct Tinker route with bounded same-session auto-continuation. |
| `inkling` / `opencode-inkling` (Mega branch) | OpenCode to OpenRouter's OpenAI-compatible API at `https://openrouter.ai/api/v1` | `OPENROUTER_API_KEY` | mega | Important alias collision: `inkling` means direct Tinker on Hard but OpenRouter on Mega. Mega defaults to high reasoning and bounded auto-continuation. |
| `deepseek-claude` | Claude Code to `DEEPSEEK_ANTHROPIC_BASE_URL` or `https://api.deepseek.com/anthropic` | `DEEPSEEK_API_KEY` | hard, cuda, mini, mega, multi | Intended for DeepSeek V4 Pro/Flash. Multi uses the fixed URL and loads the key from `~/.kbm_env`. |
| `qwen-claude` | Claude Code to `QWEN_ANTHROPIC_BASE_URL` or the token-plan endpoint `https://token-plan.ap-southeast-1.maas.aliyuncs.com/apps/anthropic` | `QWEN_API_KEY` or `DASHSCOPE_API_KEY` | hard, cuda, mini, mega | The token-plan route serves `qwen3.8-max`; a DashScope key requires the Model Studio URL override. Tool search is disabled, effort is `xhigh`, context is 983,616 tokens, and the subagent model is pinned. Mega's post-run completeness selector omits this label. |
| `codex` | Native Codex CLI to its configured OpenAI transport | CLI login or `OPENAI_API_KEY` | hard, cuda, mini, mega, multi | Forwards reasoning effort and archives the rich session JSONL by parsed session ID. |
| `kimi` | Native Kimi CLI | Kimi CLI login/config | hard, cuda, mini, mega | The branch does not pass the runner's model argument; it invokes `kimi -w ... --print`. |
| `droid` | Factory Droid CLI to its configured provider | Droid login; container may receive `FACTORY_API_KEY` and `DROID_API_KEY` | hard, cuda, mini, mega | Forwards reasoning effort. Provider endpoint depends on Droid configuration. |
| `gemini` | Native Gemini CLI | Gemini CLI login or `GEMINI_API_KEY` | hard, cuda, mini, mega | Runs from the problem directory with yolo approval. |
| `cursor` | Cursor Agent CLI (`agent`) | Cursor CLI login; container may receive `CURSOR_API_KEY` | hard, cuda, mini, mega | The executable is `agent`, not `cursor`. |
| `grok` | Native Grok CLI | Grok CLI login; optionally `XAI_API_KEY` or `GROK_API_KEY` | hard, cuda, mini, mega, multi | Uses the top-level headless command, not `grok agent`; reasoning effort is forwarded. |
| `opencode` | OpenCode to the provider/model encoded in the model argument | Provider-dependent; container forwards `OPENAI_API_KEY`, `OPENROUTER_API_KEY`, `ZAI_API_KEY`, `DEEPSEEK_API_KEY`, `MINIMAX_API_KEY`, `GEMINI_API_KEY`, and `SAKANA_API_KEY` | hard, cuda, mini, mega | Generic OpenAI-shaped route; model syntax is `provider/model`. Container mode has a stall watchdog. |
| `opencode-nemotron` | OpenCode to OpenRouter `/api/v1`, pinned to DeepInfra with fallbacks disabled | `OPENROUTER_API_KEY` | hard, cuda, mini | Preferred Nemotron route; uses an archive-local OpenCode config so the serving stack cannot drift. |
| `nvcf-nemotron` | OpenCode to a per-run localhost OpenAI adapter, then NVIDIA NVCF | One of `NGC_API_KEY`, `NVIDIA_API_KEY`, or `NVCF_API_KEY` | hard, cuda, mini | NVCF is not OpenAI-compatible directly; the runner starts the adapter. Diagnostic route. |
| `lfm-opencode` | OpenCode to local vLLM at `KBMINI_BASE_URL` or `http://127.0.0.1:8765/v1` | `KBMINI_API_KEY` (defaults to `local`) | mini (dispatchable on hard/cuda via the shared runner, but meaningful only with mini's local serving) | Uses an archive-local OpenCode config. A real secret is not normally required for the local server. |
| `lfm-claude` | Claude Code to local ccr-rust at `CCR_BASE_URL` or `http://127.0.0.1:3456`, then local vLLM | `KBMINI_API_KEY` (defaults to `local`); ccr-rust must already be running | mini | Passes the local key as `ANTHROPIC_API_KEY`. |
| `hermes` | Nous Hermes Agent to local vLLM through `OPENAI_BASE_URL=KBMINI_BASE_URL` | `KBMINI_API_KEY` (defaults to `local`) | mini | `KBMINI_HERMES_PROVIDER`, `KBMINI_HERMES_MODEL`, and `KBMINI_HERMES_MAX_TURNS` override its invocation. |
| `pi` | badlogic pi through a generated `lfm` OpenAI-completions provider to local vLLM | `KBMINI_API_KEY` (defaults to `local`) | mini | Additively updates `~/.pi/agent/models.json`; `--no-session` avoids a headless hang. |
| `lfm-grok` | Grok CLI custom `chat_completions` model to local vLLM | `KBMINI_API_KEY` (defaults to `local`) | mini | Additively appends a model block to `~/.grok/config.toml`. |
| `opencode-or` | OpenCode to OpenRouter's OpenAI-compatible API at `https://openrouter.ai/api/v1` | `OPENROUTER_API_KEY` from `~/.kbm_env` | multi | Pins `KBM_OR_PROVIDER` with fallbacks disabled. The adapter has stalled intermittently, and the branch ignores the reasoning-effort argument. |
