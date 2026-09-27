"""
VERA Verification Engine Sandbox & Dual Harness
===============================================
Executes candidate Python programs against worked examples in *separate OS
processes*:

- Every run is a ``fork()`` from a pre-warmed, stdlib-only worker interpreter
  (fork ~1 ms, no interpreter start-up cost), so dispatch overhead stays in the
  low single-digit milliseconds while the candidate can never touch the host
  process (no shared globals, no ``sys.settrace`` tricks, real ``SIGKILL`` on
  timeout).
- Isolation: temp CWD, stdin/stdout/stderr redirected to files, RLIMIT_AS /
  RLIMIT_CPU / RLIMIT_FSIZE caps, output size cap.
- Dual harness: Mode A (script: stdin piped, stdout captured) and Mode B
  (call: ``class Solution`` / ``def solve`` entrypoint invoked with arguments
  parsed from the example input). A candidate passes if either harness
  matches.
- Adaptive timeout policy: 0.75 s default, 0.30 s after >=2 timeouts,
  skipped after >=4 timeouts (per code fingerprint).
- Normalized output comparison via :mod:`vera.verify.comparator`.

The pool (:class:`VerificationSandbox`) can run many candidates in parallel
(``workers = os.cpu_count()`` by default).
"""

from __future__ import annotations

import ast
import hashlib
import io
import multiprocessing as mp
import os
import queue
import re
import select
import shutil
import signal
import sys
import tempfile
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from vera.verify.comparator import compare_outputs
from vera.verify.parser import ExamplePair

_EXC_MARKER = "VERA_EXC:"
_MAX_CAPTURE_BYTES = 1 << 20  # 1 MiB read back per stream
_DEFAULT_MEM_MB = 1024
_FSIZE_LIMIT = 32 << 20  # 32 MiB per output file


# --------------------------------------------------------------------------- #
# Result dataclasses (public API, unchanged)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ExecutionResult:
    status: str  # 'ok', 'timeout', 'timeout_skipped', 'error', 'syntax_error', 'empty_output'
    stdout: str
    stderr: str
    runtime_ms: float
    harness_mode: str  # 'script', 'call', 'none'
    matched: bool
    error_message: Optional[str] = None
    trace_values: Optional[List[str]] = None  # int/str locals seen during the run (trace mode only)


@dataclass
class CandidateVerificationResult:
    passed_examples: int
    total_examples: int
    pass_rate: float
    first_output: str
    all_passed: bool
    results: List[ExecutionResult] = field(default_factory=list)
    total_runtime_ms: float = 0.0


class AdaptiveTimeoutTracker:
    """Tracks timeouts per code fingerprint and implements adaptive demotion."""

    def __init__(self, default_timeout: float = 0.75, reduced_timeout: float = 0.30):
        self.default_timeout = default_timeout
        self.reduced_timeout = reduced_timeout
        self._counts: Dict[str, int] = {}
        self._lock = threading.Lock()

    @staticmethod
    def get_key(code: str) -> str:
        return hashlib.md5(code.encode("utf-8", errors="ignore")).hexdigest()

    def get_timeout(self, code_key: str) -> Optional[float]:
        with self._lock:
            count = self._counts.get(code_key, 0)
            if count >= 4:
                return None  # Skip execution
            if count >= 2:
                return self.reduced_timeout
            return self.default_timeout

    def record_timeout(self, code_key: str) -> None:
        with self._lock:
            self._counts[code_key] = self._counts.get(code_key, 0) + 1

    def reset(self) -> None:
        with self._lock:
            self._counts.clear()


# --------------------------------------------------------------------------- #
# Static analysis helpers (run in the parent)
# --------------------------------------------------------------------------- #
_PY2_PRINT_RE = re.compile(r"^(\s*)print\s+(?!\()(.+?)\s*$", re.MULTILINE)


def _compile_quiet(src: str) -> None:
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        compile(src, "solution.py", "exec")


def _try_compile(code: str) -> Tuple[Optional[str], Optional[str]]:
    """Return (possibly transformed code, error). Attempts a light py2 -> py3 rescue."""
    try:
        _compile_quiet(code)
        return code, None
    except SyntaxError as e:
        first_err = f"SyntaxError: {e.msg} (line {e.lineno})"

    # Cheap rescue for the most common Python-2 statements in the APPS corpus.
    rescued = code
    rescued = _PY2_PRINT_RE.sub(lambda m: f"{m.group(1)}print({m.group(2)})", rescued)
    rescued = re.sub(r"\braw_input\(", "input(", rescued)
    rescued = re.sub(r"\bxrange\(", "range(", rescued)
    rescued = re.sub(r"except\s+([A-Za-z_][\w\.]*)\s*,\s*([A-Za-z_]\w*)\s*:", r"except \1 as \2:", rescued)
    if rescued != code:
        try:
            _compile_quiet(rescued)
            return rescued, None
        except SyntaxError:
            pass

    # Module-level ``return`` / ``nonlocal`` / ``yield``: the APPS solution was stripped out of a function body.
    # Re-wrap it (hoisting __future__ imports) and try again.
    if re.search(r"outside function|no binding for nonlocal|'yield' outside", first_err):
        wrapped = _wrap_in_function(code)
        try:
            _compile_quiet(wrapped)
            return wrapped, None
        except SyntaxError:
            pass
        wrapped2 = _wrap_in_function(rescued) if rescued != code else None
        if wrapped2:
            try:
                _compile_quiet(wrapped2)
                return wrapped2, None
            except SyntaxError:
                pass

    # Last resort: lib2to3 (present up to Python 3.12; slow, so only on failure).
    try:  # pragma: no cover - environment dependent
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            from lib2to3 import refactor  # type: ignore

        tool = _get_2to3_tool(refactor)
        converted = str(tool.refactor_string(code.replace("\r\n", "\n") + "\n", "solution.py"))
        _compile_quiet(converted)
        return converted, None
    except Exception:
        return None, first_err


_2TO3_TOOL = None


def _wrap_in_function(code: str) -> str:
    """Indent the whole program into ``def __vera_main__():`` and call it (fixes module-level return/nonlocal)."""
    lines = code.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    futures = [ln for ln in lines if re.match(r"\s*from\s+__future__\s+import", ln)]
    body = [("    " + ln) if ln.strip() else ln for ln in lines if ln not in futures]
    head = "\n".join(f.strip() for f in futures)
    return (head + "\n" if head else "") + "def __vera_main__():\n" + "\n".join(body) + "\n    return\n\n__vera_main__()\n"


def _get_2to3_tool(refactor):  # pragma: no cover - environment dependent
    global _2TO3_TOOL
    if _2TO3_TOOL is None:
        fixers = refactor.get_fixers_from_package("lib2to3.fixes")
        _2TO3_TOOL = refactor.RefactoringTool(fixers, options={"print_function": False})
    return _2TO3_TOOL


def detect_call_entry(code: str) -> Optional[Dict[str, Any]]:
    """Find a Mode-B entrypoint: LeetCode ``class Solution`` method or a top-level solve()/main()."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return None

    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name in ("Solution", "Solver"):
            methods = [item for item in node.body if isinstance(item, ast.FunctionDef) and not item.name.startswith("_")]
            if not methods:
                continue
            # Helpers are referenced as self.<name>( inside the class; the entrypoint usually is not.
            referenced = set()
            for item in ast.walk(node):
                if isinstance(item, ast.Call) and isinstance(item.func, ast.Attribute):
                    if isinstance(item.func.value, ast.Name) and item.func.value.id == "self":
                        referenced.add(item.func.attr)
            ordered = [m for m in methods if m.name not in referenced] + [m for m in methods if m.name in referenced]
            return {
                "kind": "class",
                "cls": node.name,
                "methods": [(m.name, [a.arg for a in m.args.args if a.arg not in ("self", "cls")]) for m in ordered],
            }
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name.lower() in ("solve", "solution", "main"):
            params = [a.arg for a in node.args.args]
            return {"kind": "func", "func": node.name, "params": params}
    top_funcs = [n for n in tree.body if isinstance(n, ast.FunctionDef) and not n.name.startswith("_")]
    if len(top_funcs) == 1 and top_funcs[0].name != "__vera_main__":
        fn = top_funcs[0]
        return {"kind": "func", "func": fn.name, "params": [a.arg for a in fn.args.args]}
    return None


# --------------------------------------------------------------------------- #
# Code that runs *inside the forked child*
# --------------------------------------------------------------------------- #
_JSON_LITERAL_FIX = (("true", "True"), ("false", "False"), ("null", "None"))


def _parse_call_args(stdin_text: str) -> Tuple[List[Any], Dict[str, Any]]:
    """Parse ``nums = [2,7,11,15], target = 9`` or ``[2,7] 9`` into (args, kwargs)."""
    text = stdin_text.strip()
    for src, dst in _JSON_LITERAL_FIX:
        text = re.sub(rf"\b{src}\b", dst, text)

    kv_positions = [m for m in re.finditer(r"(?:^|[,;\s])([A-Za-z_]\w*)\s*[=:]\s*(?![=:])", text)]
    if kv_positions:
        kwargs: Dict[str, Any] = {}
        for i, m in enumerate(kv_positions):
            start = m.end()
            end = kv_positions[i + 1].start() if i + 1 < len(kv_positions) else len(text)
            raw = text[start:end].strip().rstrip(",").strip()
            try:
                kwargs[m.group(1)] = ast.literal_eval(raw)
            except Exception:
                kwargs[m.group(1)] = raw.strip("\"'")
        return [], kwargs

    # Positional: comma/newline separated literals.
    candidates = [text, "[" + text + "]", "[" + text.replace("\n", ",") + "]"]
    for cand in candidates:
        try:
            val = ast.literal_eval(cand)
            if isinstance(val, list) and cand != text:
                return list(val), {}
            return [val], {}
        except Exception:
            continue
    toks = text.split()
    out: List[Any] = []
    for tok in toks:
        try:
            out.append(ast.literal_eval(tok))
        except Exception:
            out.append(tok)
    return out, {}


def _format_return(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, float):
        return repr(value)
    return repr(value).replace(" ", "")


def _run_call_mode(g: Dict[str, Any], entry: Dict[str, Any], stdin_text: str) -> None:
    import inspect

    args, kwargs = _parse_call_args(stdin_text)

    if entry["kind"] == "class":
        inst = g[entry["cls"]]()
        candidates = [getattr(inst, name) for name, _ in entry["methods"] if hasattr(inst, name)]
    else:
        candidates = [g[entry["func"]]]

    def _bind(fn):
        try:
            sig = inspect.signature(fn)
        except (TypeError, ValueError):
            return None
        for attempt in ((), (kwargs,), (list(kwargs.values()),), (args,)):
            try:
                if attempt == ():
                    if kwargs or args:
                        continue
                    sig.bind()
                    return lambda: fn()
                if isinstance(attempt[0], dict):
                    if not kwargs:
                        continue
                    sig.bind(**attempt[0])
                    return lambda: fn(**attempt[0])
                if not attempt[0]:
                    continue
                sig.bind(*attempt[0])
                return lambda a=attempt[0]: fn(*a)
            except TypeError:
                continue
        return None

    call = None
    for fn in candidates:
        call = _bind(fn)
        if call is not None:
            break
    if call is None:
        if not candidates:
            raise TypeError("no callable entrypoint")
        call = lambda: candidates[0](*args, **kwargs)  # noqa: E731
    ret = call()
    sys.stdout.write(_format_return(ret))
    sys.stdout.write("\n")


_TRACE_MAX_VALUES = 4000


def _make_value_tracer(sink: set):
    """sys.settrace hook collecting int/str local values on 'line'/'return' events (bounded)."""
    def tracer(frame, event, arg):
        if frame.f_code.co_filename != "solution.py":
            return None
        if event in ("line", "return"):
            for v in frame.f_locals.values():
                if isinstance(v, bool):
                    continue
                if isinstance(v, int) and abs(v) < 10**15:
                    sink.add(str(v))
                elif isinstance(v, str) and 0 < len(v) <= 64:
                    sink.add(v)
                elif isinstance(v, (list, tuple)) and len(v) <= 64:
                    for x in v:
                        if isinstance(x, int) and not isinstance(x, bool) and abs(x) < 10**15:
                            sink.add(str(x))
            if len(sink) > _TRACE_MAX_VALUES:
                sys.settrace(None)
                return None
        return tracer
    return tracer


def _child_main(
    code: str,
    stdin_text: str,
    paths: Tuple[str, str, str],
    cwd: str,
    mem_mb: int,
    cpu_seconds: int,
    call_entry: Optional[Dict[str, Any]],
    trace_values: bool = False,
) -> None:  # pragma: no cover - runs in forked child
    """Never returns: ends with os._exit()."""
    import builtins
    import json as _json
    import resource

    stdin_path, stdout_path, stderr_path = paths
    exit_code = 0
    trace_sink: set = set()
    try:
        os.chdir(cwd)
        mem_bytes = mem_mb << 20
        for res_name, limit in (
            ("RLIMIT_AS", (mem_bytes, mem_bytes)),
            ("RLIMIT_CPU", (cpu_seconds, cpu_seconds + 1)),
            ("RLIMIT_FSIZE", (_FSIZE_LIMIT, _FSIZE_LIMIT)),
            ("RLIMIT_CORE", (0, 0)),
        ):
            try:
                resource.setrlimit(getattr(resource, res_name), limit)
            except (ValueError, OSError):
                pass
        try:  # allow deep recursion (many APPS solutions raise the recursion limit)
            soft, hard = resource.getrlimit(resource.RLIMIT_STACK)
            want = 512 << 20
            if hard == resource.RLIM_INFINITY or hard >= want:
                resource.setrlimit(resource.RLIMIT_STACK, (want, hard))
        except (ValueError, OSError):
            pass

        fd_in = os.open(stdin_path, os.O_RDONLY)
        fd_out = os.open(stdout_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        fd_err = os.open(stderr_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        os.dup2(fd_in, 0)
        os.dup2(fd_out, 1)
        os.dup2(fd_err, 2)
        for fd in (fd_in, fd_out, fd_err):
            if fd > 2:
                os.close(fd)

        sys.stdin = io.TextIOWrapper(io.FileIO(0, "r", closefd=False), encoding="utf-8", errors="replace")
        sys.stdout = io.TextIOWrapper(io.FileIO(1, "w", closefd=False), encoding="utf-8", errors="replace")
        sys.stderr = io.TextIOWrapper(io.FileIO(2, "w", closefd=False), encoding="utf-8", errors="replace")
        sys.__stdin__, sys.__stdout__, sys.__stderr__ = sys.stdin, sys.stdout, sys.stderr
        sys.argv = ["solution.py"]
        sys.setrecursionlimit(20000)

        g: Dict[str, Any] = {"__name__": "__main__", "__builtins__": builtins, "__file__": "solution.py"}
        # Common py2-isms and LeetCode starter-code names that solutions assume exist.
        g.setdefault("raw_input", input)
        g.setdefault("xrange", range)
        g.setdefault("unicode", str)
        try:
            import bisect
            import collections
            import functools
            import heapq
            import itertools
            import math
            import operator
            import random
            import string
            import typing

            for name in ("List", "Dict", "Set", "Tuple", "Optional", "Union", "Any", "Deque", "DefaultDict",
                         "Iterable", "Iterator", "Callable", "FrozenSet", "Sequence", "Mapping"):
                g.setdefault(name, getattr(typing, name))
            for mod in (math, heapq, collections, itertools, bisect, functools, string, re, operator, random, sys, os):
                g.setdefault(mod.__name__, mod)
            for name, val in (
                ("defaultdict", collections.defaultdict), ("deque", collections.deque), ("Counter", collections.Counter),
                ("OrderedDict", collections.OrderedDict), ("heappush", heapq.heappush), ("heappop", heapq.heappop),
                ("heapify", heapq.heapify), ("lru_cache", functools.lru_cache), ("cache", getattr(functools, "cache", functools.lru_cache)),
                ("reduce", functools.reduce), ("inf", math.inf), ("gcd", math.gcd), ("sqrt", math.sqrt),
                ("bisect_left", bisect.bisect_left), ("bisect_right", bisect.bisect_right), ("insort", bisect.insort),
                ("accumulate", itertools.accumulate), ("permutations", itertools.permutations),
                ("combinations", itertools.combinations), ("product", itertools.product),
            ):
                g.setdefault(name, val)
        except Exception:
            pass

        import warnings

        warnings.simplefilter("ignore")
        compiled = compile(code, "solution.py", "exec")
        if trace_values:
            sys.settrace(_make_value_tracer(trace_sink))
        try:
            exec(compiled, g)
        except SystemExit as se:  # exit()/quit() after printing is normal in CP code
            if se.code not in (0, None):
                exit_code = 3
                sys.stderr.write(f"\n{_EXC_MARKER}SystemExit: {se.code}\n")

        if trace_values:
            sys.settrace(None)
            try:
                with open(os.path.join(cwd, "trace.json"), "w", encoding="utf-8") as tf:
                    tf.write(_json.dumps(sorted(trace_sink)))
            except Exception:
                pass
        # Wait for non-daemon threads the solution may have started (threading.Thread(target=main).start()).
        main_thread = threading.main_thread()
        for t in threading.enumerate():
            if t is not main_thread and not t.daemon:
                t.join()

        sys.stdout.flush()
        if exit_code == 0 and call_entry is not None and os.fstat(1).st_size == 0:
            try:
                _run_call_mode(g, call_entry, stdin_text)
                sys.stdout.flush()
                sys.stderr.write("\nVERA_MODE:call\n")
            except BaseException as exc:  # noqa: BLE001
                sys.stderr.write(f"\n{_EXC_MARKER}{type(exc).__name__}: {str(exc)[:200]}\n")
                exit_code = 1
    except BaseException as exc:  # noqa: BLE001
        try:
            sys.stdout.flush()
        except Exception:
            pass
        try:
            sys.stderr.write(f"\n{_EXC_MARKER}{type(exc).__name__}: {str(exc)[:200]}\n")
        except Exception:
            pass
        exit_code = 1
    finally:
        try:
            sys.stdout.flush()
            sys.stderr.flush()
        except Exception:
            pass
        os._exit(exit_code)


def _read_capped(path: str, cap: int = _MAX_CAPTURE_BYTES) -> str:
    try:
        with open(path, "rb") as f:
            data = f.read(cap + 1)
    except OSError:
        return ""
    if len(data) > cap:
        data = data[:cap]
    return data.decode("utf-8", errors="replace")


def run_isolated(
    code: str,
    stdin_text: str,
    timeout: float,
    mem_mb: int = _DEFAULT_MEM_MB,
    call_entry: Optional[Dict[str, Any]] = None,
    workdir: Optional[str] = None,
    trace_values: bool = False,
) -> Dict[str, Any]:
    """Fork the current process and execute ``code`` in the child under a wall-clock timeout.

    Returns a plain dict (picklable): status, stdout, stderr, runtime_ms, mode, error.
    Meant to be called from a single-threaded worker process (see :class:`_WorkerPool`),
    but works from any process on POSIX.
    """
    tmp = tempfile.mkdtemp(prefix="vera_", dir=workdir)
    stdin_path = os.path.join(tmp, "in.txt")
    stdout_path = os.path.join(tmp, "out.txt")
    stderr_path = os.path.join(tmp, "err.txt")
    with open(stdin_path, "w", encoding="utf-8") as f:
        f.write(stdin_text)

    cpu_seconds = max(1, int(timeout) + 1)
    r_fd, w_fd = os.pipe()
    t0 = time.perf_counter()
    pid = os.fork()
    if pid == 0:  # child
        os.close(r_fd)
        # w_fd is inherited and closes automatically when the child exits -> parent sees EOF.
        _child_main(code, stdin_text, (stdin_path, stdout_path, stderr_path), tmp, mem_mb, cpu_seconds, call_entry,
                    trace_values=trace_values)
        os._exit(1)  # unreachable

    os.close(w_fd)
    timed_out = False
    try:
        ready, _, _ = select.select([r_fd], [], [], timeout)
        if not ready:
            timed_out = True
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
    finally:
        os.close(r_fd)
    _, status = os.waitpid(pid, 0)
    runtime_ms = (time.perf_counter() - t0) * 1000.0

    stdout = _read_capped(stdout_path)
    stderr = _read_capped(stderr_path)
    traced: Optional[List[str]] = None
    if trace_values:
        try:
            import json as _json

            with open(os.path.join(tmp, "trace.json"), "r", encoding="utf-8") as tf:
                traced = _json.loads(tf.read())
        except Exception:
            traced = []
    shutil.rmtree(tmp, ignore_errors=True)

    mode = "call" if "VERA_MODE:call" in stderr else "script"
    error: Optional[str] = None
    if timed_out:
        result_status = "timeout"
        error = f"Timeout after {runtime_ms:.1f}ms"
    elif os.WIFSIGNALED(status):
        result_status = "error"
        error = f"Killed by signal {os.WTERMSIG(status)}"
    else:
        code_ = os.WEXITSTATUS(status)
        if code_ == 0:
            result_status = "ok" if stdout.strip() else "empty_output"
        else:
            result_status = "error"
            m = re.findall(rf"{_EXC_MARKER}(.*)", stderr)
            error = m[-1].strip() if m else f"exit code {code_}"
    # Strip our own markers out of stderr before returning.
    stderr = re.sub(rf"\n?{_EXC_MARKER}.*", "", stderr)
    stderr = stderr.replace("\nVERA_MODE:call\n", "")
    return {
        "status": result_status,
        "stdout": stdout,
        "stderr": stderr,
        "runtime_ms": runtime_ms,
        "mode": mode if stdout.strip() else "none",
        "error": error,
        "trace_values": traced,
    }


# --------------------------------------------------------------------------- #
# Worker pool: pre-warmed stdlib-only interpreters that fork per run
# --------------------------------------------------------------------------- #
# Workers are plain ``python -c`` subprocesses (not multiprocessing), so they never re-import
# the caller's __main__, carry none of the parent's threads (torch/OpenMP) and are therefore
# safe to fork() from. Frames on the pipe: 4-byte big-endian length + pickle payload.
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_WORKER_BOOT = "import sys; sys.path.insert(0, %r); from vera.verify.executor import _worker_stdio_loop; _worker_stdio_loop()"


def _read_exact(fd: int, n: int) -> bytes:
    chunks: List[bytes] = []
    remaining = n
    while remaining > 0:
        chunk = os.read(fd, remaining)
        if not chunk:
            raise EOFError
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def _send_frame(fd: int, obj: Any) -> None:
    import pickle

    payload = pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
    data = len(payload).to_bytes(4, "big") + payload
    view = memoryview(data)
    while view:
        written = os.write(fd, view)
        view = view[written:]


def _recv_frame(fd: int) -> Any:
    import pickle

    header = _read_exact(fd, 4)
    length = int.from_bytes(header, "big")
    return pickle.loads(_read_exact(fd, length))


def _worker_stdio_loop() -> None:  # pragma: no cover - separate process
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    in_fd, out_fd = 0, 1
    while True:
        try:
            job = _recv_frame(in_fd)
        except (EOFError, OSError):
            break
        if job is None:
            break
        try:
            res = run_isolated(**job)
        except Exception as exc:  # noqa: BLE001
            res = {
                "status": "error", "stdout": "", "stderr": traceback.format_exc()[-2000:],
                "runtime_ms": 0.0, "mode": "none", "error": f"WorkerError: {exc}",
            }
        try:
            _send_frame(out_fd, res)
        except (OSError, BrokenPipeError):
            break


class _Worker:
    def __init__(self):
        import subprocess

        env = dict(os.environ)
        env["PYTHONPATH"] = _PROJECT_ROOT + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
        env.setdefault("PYTHONHASHSEED", "0")
        self.proc = subprocess.Popen(
            [sys.executable, "-c", _WORKER_BOOT % _PROJECT_ROOT],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            env=env, close_fds=True,
        )
        self.in_fd = self.proc.stdin.fileno()  # type: ignore[union-attr]
        self.out_fd = self.proc.stdout.fileno()  # type: ignore[union-attr]

    def run(self, job: Dict[str, Any]) -> Dict[str, Any]:
        _send_frame(self.in_fd, job)
        return _recv_frame(self.out_fd)

    def alive(self) -> bool:
        return self.proc.poll() is None

    def close(self) -> None:
        try:
            if self.alive():
                _send_frame(self.in_fd, None)
        except Exception:
            pass
        try:
            self.proc.stdin.close()  # type: ignore[union-attr]
            self.proc.stdout.close()  # type: ignore[union-attr]
        except Exception:
            pass
        try:
            self.proc.wait(timeout=1.0)
        except Exception:
            try:
                self.proc.kill()
            except Exception:
                pass


class _WorkerPool:
    """N long-lived worker processes; each job is forwarded over a pipe and run via fork()."""

    def __init__(self, workers: int):
        self.workers = max(1, workers)
        self._idle: "queue.Queue[_Worker]" = queue.Queue()
        self._all: List[_Worker] = []
        self._lock = threading.Lock()
        self._closed = False
        for _ in range(self.workers):
            self._start_one()

    def _start_one(self) -> _Worker:
        w = _Worker()
        with self._lock:
            self._all.append(w)
        self._idle.put(w)
        return w

    def run(self, job: Dict[str, Any]) -> Dict[str, Any]:
        w = self._idle.get()
        try:
            return w.run(job)
        except (EOFError, OSError, BrokenPipeError):
            # Worker died mid-job; replace it and retry the job once on a fresh worker.
            with self._lock:
                if w in self._all:
                    self._all.remove(w)
            w.close()
            w = _Worker()
            with self._lock:
                self._all.append(w)
            try:
                return w.run(job)
            except (EOFError, OSError, BrokenPipeError):
                return {"status": "error", "stdout": "", "stderr": "", "runtime_ms": 0.0, "mode": "none",
                        "error": "WorkerCrashed"}
        finally:
            self._idle.put(w)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        with self._lock:
            workers = list(self._all)
            self._all.clear()
        for w in workers:
            w.close()

    def __del__(self):  # pragma: no cover
        try:
            self.close()
        except Exception:
            pass


# --------------------------------------------------------------------------- #
# Public sandbox
# --------------------------------------------------------------------------- #
class VerificationSandbox:
    """Process-isolated execution sandbox with dual harness, adaptive timeouts and a worker pool.

    Parameters
    ----------
    default_timeout, reduced_timeout : float
        Wall-clock timeouts (seconds) for the adaptive policy.
    workers : int | None
        Number of pre-warmed worker processes (default ``os.cpu_count()``). ``0`` disables the
        pool and forks directly from the calling process (fine for tests / single-threaded use).
    mem_limit_mb : int
        RLIMIT_AS for each candidate process.
    """

    def __init__(
        self,
        default_timeout: float = 0.75,
        reduced_timeout: float = 0.30,
        workers: Optional[int] = None,
        mem_limit_mb: int = _DEFAULT_MEM_MB,
    ):
        self.timeout_tracker = AdaptiveTimeoutTracker(default_timeout, reduced_timeout)
        self.mem_limit_mb = mem_limit_mb
        self.workers = os.cpu_count() or 2 if workers is None else workers
        self._pool: Optional[_WorkerPool] = None
        self._pool_lock = threading.Lock()

    # ----- pool management -------------------------------------------------- #
    def _get_pool(self) -> Optional[_WorkerPool]:
        if self.workers <= 0:
            return None
        if self._pool is None:
            with self._pool_lock:
                if self._pool is None:
                    self._pool = _WorkerPool(self.workers)
        return self._pool

    def close(self) -> None:
        if self._pool is not None:
            self._pool.close()
            self._pool = None

    def __enter__(self) -> "VerificationSandbox":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # ----- single run ------------------------------------------------------- #
    def _dispatch(self, job: Dict[str, Any]) -> Dict[str, Any]:
        pool = self._get_pool()
        if pool is None:
            return run_isolated(**job)
        return pool.run(job)

    def execute_snippet(
        self,
        code: str,
        stdin_str: str,
        expected_output: Optional[str] = None,
        force_timeout: Optional[float] = None,
        call_entry: Optional[Dict[str, Any]] = None,
        multiline_set: bool = False,
        trace_values: bool = False,
    ) -> ExecutionResult:
        """Run candidate code once through the dual harness and compare with ``expected_output``."""
        code_key = self.timeout_tracker.get_key(code)
        timeout = force_timeout if force_timeout is not None else self.timeout_tracker.get_timeout(code_key)
        if timeout is None:
            return ExecutionResult(
                status="timeout_skipped", stdout="", stderr="Skipped due to repeated timeouts (>=4)",
                runtime_ms=0.0, harness_mode="none", matched=False, error_message="adaptive_timeout_skipped",
            )

        runnable, syntax_err = _try_compile(code)
        if runnable is None:
            return ExecutionResult(
                status="syntax_error", stdout="", stderr=syntax_err or "", runtime_ms=0.0,
                harness_mode="none", matched=False, error_message=syntax_err,
            )
        if call_entry is None:
            call_entry = detect_call_entry(runnable)

        raw = self._dispatch({
            "code": runnable,
            "stdin_text": stdin_str,
            "timeout": float(timeout),
            "mem_mb": self.mem_limit_mb,
            "call_entry": call_entry,
            "trace_values": trace_values,
        })

        if raw["status"] == "timeout":
            self.timeout_tracker.record_timeout(code_key)

        matched = False
        if expected_output is not None and raw["stdout"].strip():
            matched = compare_outputs(raw["stdout"], expected_output, multiline_set=multiline_set)

        status = raw["status"]
        if status == "ok" and expected_output is not None and not matched:
            status = "ok"  # ran fine, just wrong answer; caller checks .matched
        return ExecutionResult(
            status=status,
            stdout=raw["stdout"],
            stderr=raw["stderr"],
            runtime_ms=raw["runtime_ms"],
            harness_mode=raw["mode"],
            matched=matched,
            error_message=raw["error"],
            trace_values=raw.get("trace_values"),
        )

    # ----- candidate over all examples ------------------------------------- #
    def verify_candidate(
        self,
        code: str,
        examples: Sequence[ExamplePair],
        multiline_set: bool = False,
    ) -> CandidateVerificationResult:
        """Run candidate program against all extracted example pairs (stops early on crash/timeout)."""
        if not examples:
            return CandidateVerificationResult(0, 0, 0.0, "", False, [], 0.0)

        runnable, syntax_err = _try_compile(code)
        call_entry = detect_call_entry(runnable) if runnable is not None else None

        passed = 0
        results: List[ExecutionResult] = []
        total_time = 0.0
        first_output = ""
        for idx, ex in enumerate(examples):
            if runnable is None:
                res = ExecutionResult("syntax_error", "", syntax_err or "", 0.0, "none", False, syntax_err)
            else:
                res = self.execute_snippet(
                    code=runnable, stdin_str=ex.stdin, expected_output=ex.expected_stdout,
                    call_entry=call_entry, multiline_set=multiline_set,
                )
                # Statement formatting often inserts blank lines between input lines (CodeChef/AtCoder);
                # if the run did not match and the stdin has blank lines, retry once with them removed.
                if not res.matched and res.status != "timeout_skipped" and "\n\n" in ex.stdin.strip("\n"):
                    collapsed = "\n".join(ln for ln in ex.stdin.split("\n") if ln.strip()) + "\n"
                    res2 = self.execute_snippet(
                        code=runnable, stdin_str=collapsed, expected_output=ex.expected_stdout,
                        call_entry=call_entry, multiline_set=multiline_set,
                    )
                    if res2.matched or (res.status in ("error", "empty_output") and res2.status == "ok"):
                        res = res2
            results.append(res)
            total_time += res.runtime_ms
            if idx == 0:
                first_output = res.stdout.strip()
            if res.matched:
                passed += 1
            elif res.status in ("timeout", "timeout_skipped", "error", "syntax_error"):
                break  # broken program: don't burn time on the remaining examples

        n = len(examples)
        return CandidateVerificationResult(
            passed_examples=passed,
            total_examples=n,
            pass_rate=passed / n,
            first_output=first_output,
            all_passed=passed == n,
            results=results,
            total_runtime_ms=total_time,
        )

    def verify_many(
        self,
        candidates: Iterable[Tuple[str, str]],
        examples: Sequence[ExamplePair],
        multiline_set: bool = False,
    ) -> Dict[str, CandidateVerificationResult]:
        """Verify ``[(doc_id, code), ...]`` in parallel across the worker pool."""
        items = list(candidates)
        if not items:
            return {}
        if self.workers <= 0 or len(items) == 1:
            return {doc_id: self.verify_candidate(code, examples, multiline_set) for doc_id, code in items}
        self._get_pool()
        out: Dict[str, CandidateVerificationResult] = {}
        with ThreadPoolExecutor(max_workers=self.workers) as ex:
            futures = {ex.submit(self.verify_candidate, code, examples, multiline_set): doc_id for doc_id, code in items}
            for fut, doc_id in futures.items():
                out[doc_id] = fut.result()
        return out


# API Alias
ExecutionSandbox = VerificationSandbox
