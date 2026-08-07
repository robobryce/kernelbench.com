"""Run a grading script with lexical tripwires for known early-exit tricks.

Checkers and benchmarks execute ``solution.py`` in-process.  A solution that
raises ``SystemExit(0)`` would otherwise make Python report success before the
trusted script reaches its final PASS or score emission. Normal fallthrough is
the only successful completion. Source inspection and runtime shims reject
known process-replacement, termination, and frame-walking forms. These checks
are defense-in-depth tripwires, not a Python authority or security boundary;
candidate code still shares the grading interpreter and sufficiently dynamic
code can evade lexical rules. A stronger authority boundary would require
separating candidate and checker by process or sandbox.
"""

from __future__ import annotations

import ast
import os
import re
import stat
import sys
from pathlib import Path

_TRUSTED_PROBLEM_FILES = frozenset(
    {
        "PROMPT.txt",
        "benchmark.py",
        "problem.yaml",
        "reference.py",
        "shapes.py",
        "sota.py",
        "check.py",
    }
)
_RESERVED_MODULES = frozenset({"benchmark", "check", "reference", "shapes", "sota", "src"})
_SOURCE_SUFFIXES = frozenset({".c", ".cc", ".cpp", ".cu", ".cuh", ".h", ".hpp", ".py"})
_BLOCKED_ATTRIBUTES = frozenset(
    {
        "__code__",
        "__globals__",
        "_exit",
        "_getframe",
        "chdir",
        "currentframe",
        "execl",
        "execle",
        "execlp",
        "execlpe",
        "execv",
        "execve",
        "execvp",
        "execvpe",
        "f_back",
        "fchdir",
        "f_globals",
        "f_locals",
        "exit",
        "spawnl",
        "spawnle",
        "spawnlp",
        "spawnlpe",
        "spawnv",
        "spawnve",
        "spawnvp",
        "spawnvpe",
        "syscall",
    }
)
_BLOCKED_NATIVE = re.compile(
    rb"\b(?:_exit|exit_group|execv|execve|execvp|execvpe|execl|execle|"
    rb"execlp|execlpe|syscall)\s*\(|"
    rb"\b(?:exit|quick_exit|_Exit|Py_Exit)\s*\(\s*0\b|"
    rb"\b(?:SYS_exit|SYS_exit_group|__NR_exit|__NR_exit_group)\b"
)
_MAX_CANDIDATE_FILES = 2_048
_MAX_CANDIDATE_SOURCE_BYTES = 128 * 1024 * 1024


class CandidateControlFlowError(RuntimeError):
    """Candidate source matched a grading control-flow tripwire."""


def _constant_string(node: ast.AST) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        left = _constant_string(node.left)
        right = _constant_string(node.right)
        if left is not None and right is not None:
            return left + right
    return None


class _CandidatePolicy(ast.NodeVisitor):
    def __init__(self) -> None:
        self._sys_aliases = {"sys"}
        self._inspect_aliases = {"inspect"}
        self._inspect_stack_aliases: set[str] = set()

    def fail(self, node: ast.AST, detail: str) -> None:
        raise CandidateControlFlowError(
            f"forbidden candidate control flow at line {getattr(node, 'lineno', '?')}: {detail}"
        )

    def visit_Import(self, node: ast.Import) -> None:
        if any(alias.name == "__main__" for alias in node.names):
            self.fail(node, "import __main__")
        for alias in node.names:
            if alias.name == "sys":
                self._sys_aliases.add(alias.asname or "sys")
            if alias.name == "inspect":
                self._inspect_aliases.add(alias.asname or "inspect")
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        if node.module == "__main__":
            self.fail(node, "import from __main__")
        if node.module == "sys" and any(alias.name == "modules" for alias in node.names):
            self.fail(node, "access sys.modules")
        if node.module == "inspect":
            for alias in node.names:
                if alias.name == "stack":
                    self._inspect_stack_aliases.add(alias.asname or "stack")
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        if node.attr in _BLOCKED_ATTRIBUTES:
            self.fail(node, f"attribute {node.attr}")
        if (
            node.attr == "stack"
            and isinstance(node.value, ast.Name)
            and node.value.id in self._inspect_aliases
        ):
            self.fail(node, "inspect.stack")
        if (
            node.attr == "modules"
            and isinstance(node.value, ast.Name)
            and node.value.id in self._sys_aliases
        ):
            self.fail(node, "access sys.modules")
        self.generic_visit(node)

    def visit_Subscript(self, node: ast.Subscript) -> None:
        attribute = _constant_string(node.slice)
        if attribute in _BLOCKED_ATTRIBUTES:
            self.fail(node, f"dynamic attribute {attribute}")
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        name = node.func.id if isinstance(node.func, ast.Name) else None
        if name in {"_exit", "exit", "exit_group", "quick_exit", "syscall"}:
            self.fail(node, f"call {name}")
        if name in self._inspect_stack_aliases:
            self.fail(node, "inspect.stack")
        if name in {"getattr", "setattr", "delattr"} and len(node.args) >= 2:
            attribute = _constant_string(node.args[1])
            if attribute in _BLOCKED_ATTRIBUTES:
                self.fail(node, f"dynamic attribute {attribute}")
            if (
                attribute == "stack"
                and isinstance(node.args[0], ast.Name)
                and node.args[0].id in self._inspect_aliases
            ):
                self.fail(node, "inspect.stack")
        if name == "__import__":
            self.fail(node, "dynamic import")
        self.generic_visit(node)


def _validate_candidate_sources(problem_dir: Path) -> None:
    count = 0
    total = 0
    for path in sorted(problem_dir.rglob("*")):
        relative = path.relative_to(problem_dir)
        if any(part in {".venv", "__pycache__"} for part in relative.parts):
            continue
        metadata = path.lstat()
        if stat.S_ISLNK(metadata.st_mode):
            raise CandidateControlFlowError(f"candidate link is forbidden: {relative}")
        if stat.S_ISDIR(metadata.st_mode):
            if len(relative.parts) == 1 and (
                relative.name == "solution" or relative.name in _RESERVED_MODULES
            ):
                raise CandidateControlFlowError(
                    f"candidate shadows trusted module: {relative.name}"
                )
            continue
        if not stat.S_ISREG(metadata.st_mode):
            raise CandidateControlFlowError(f"candidate special file is forbidden: {relative}")
        if len(relative.parts) == 1 and relative.name in _TRUSTED_PROBLEM_FILES:
            continue
        top_level = relative.parts[0]
        if relative.as_posix() != "solution.py" and any(
            top_level == module or top_level.startswith(f"{module}.")
            for module in _RESERVED_MODULES
        ):
            raise CandidateControlFlowError(f"candidate shadows trusted module: {top_level}")
        with path.open("rb") as stream:
            header = stream.read(4)
        if header == b"\x7fELF":
            raise CandidateControlFlowError(
                f"precompiled candidate binary is forbidden: {relative}"
            )
        if path.suffix.lower() not in _SOURCE_SUFFIXES:
            continue
        count += 1
        total += metadata.st_size
        if count > _MAX_CANDIDATE_FILES or total > _MAX_CANDIDATE_SOURCE_BYTES:
            raise CandidateControlFlowError("candidate source exceeds inspection limits")
        source = path.read_bytes()
        if _BLOCKED_NATIVE.search(source):
            raise CandidateControlFlowError(
                f"process-level termination or replacement is forbidden: {relative}"
            )
        if path.suffix.lower() == ".py":
            try:
                tree = ast.parse(source, filename=str(relative))
            except (SyntaxError, ValueError) as exc:
                raise CandidateControlFlowError(
                    f"candidate Python source is invalid: {relative}: {exc}"
                ) from exc
            _CandidatePolicy().visit(tree)


def _block_process_replacement(module: object) -> dict[str, object]:
    def guarded(*_args: object, **_kwargs: object) -> None:
        raise CandidateControlFlowError("process termination or replacement is forbidden")

    originals: dict[str, object] = {}
    for name in _BLOCKED_ATTRIBUTES:
        if name in {"_exit", "chdir", "fchdir"} or name.startswith(("exec", "spawn")):
            if hasattr(module, name):
                originals[name] = getattr(module, name)
                setattr(module, name, guarded)
    return originals


def _restore_process_replacement(module: object, originals: dict[str, object]) -> None:
    for name, value in originals.items():
        setattr(module, name, value)


def run_grading_script(script: str | Path) -> int:
    """Run *script* as ``__main__`` and return zero only on fallthrough."""
    script_path = Path(script).resolve()
    trusted_sys = sys
    trusted_os = os
    trusted_print = print
    trusted_system_exit = SystemExit
    trusted_stderr = trusted_sys.stderr
    trusted_posix = trusted_sys.modules.get("posix")
    original_path = trusted_sys.path.copy()
    original_argv0 = trusted_sys.argv[0]
    _validate_candidate_sources(script_path.parent)

    # Use an unregistered globals dict to frustrate the direct ``__main__``
    # rebinding seen in audits. This is a tripwire, not interpreter isolation.
    script_globals = {
        "__name__": "__main__",
        "__file__": str(script_path),
        "__cached__": None,
        "__loader__": None,
        "__package__": None,
        "__spec__": None,
    }
    # Put candidate sidecars after canonical environment and repository paths;
    # the source scan separately rejects straightforward reserved-name shadows.
    trusted_sys.path.append(str(script_path.parent))
    trusted_sys.argv[0] = str(script_path)
    original_os_process_functions = _block_process_replacement(trusted_os)
    original_posix_process_functions = (
        _block_process_replacement(trusted_posix) if trusted_posix is not None else {}
    )
    try:
        source = script_path.read_bytes()
        exec(compile(source, str(script_path), "exec"), script_globals)
    except trusted_system_exit as exc:
        if exc.code not in (None, 0):
            raise
        trusted_print(
            f"FAIL: {script_path.name} exited successfully before normal completion",
            file=trusted_stderr,
            flush=True,
        )
        return 1
    finally:
        _restore_process_replacement(trusted_os, original_os_process_functions)
        if trusted_posix is not None:
            _restore_process_replacement(trusted_posix, original_posix_process_functions)
        trusted_sys.argv[0] = original_argv0
        trusted_sys.path[:] = original_path
    return 0


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    if len(args) != 1:
        print("usage: trusted_entrypoint.py <check.py|benchmark.py>", file=sys.stderr)
        return 2
    return run_grading_script(args[0])


if __name__ == "__main__":
    raise SystemExit(main())
