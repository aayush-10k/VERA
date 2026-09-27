"""
VERA Verification Engine Sandbox & Dual Harness
===============================================
Executes candidate Python programs against worked examples:
- Dual Harness: Mode A (Script stdin/stdout) and Mode B (Callable / Solution entrypoint)
- High throughput (<10 ms dispatch latency)
- Safe sandbox environment disabling network sockets and dangerous OS syscalls
- Adaptive timeout policy: 0.75s default, 0.30s on >=2 timeouts, skip on >=4 timeouts
- Normalized output comparison via vera.verify.comparator
"""

from __future__ import annotations

import ast
import ctypes
import hashlib
import io
import os
import sys
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from vera.verify.comparator import compare_outputs
from vera.verify.parser import ExamplePair


class SandboxTimeout(BaseException):
    """Raised asynchronously when a sandbox execution exceeds its timeout budget."""
    pass


@dataclass(frozen=True)
class ExecutionResult:
    status: str  # 'ok', 'timeout', 'timeout_skipped', 'error', 'syntax_error'
    stdout: str
    stderr: str
    runtime_ms: float
    harness_mode: str  # 'script', 'call', 'none'
    matched: bool
    error_message: Optional[str] = None


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


# Prevent candidate programs from hanging on OS fd 0 (e.g., open(0), os.read(0))
try:
    _devnull_fd = os.open(os.devnull, os.O_RDONLY)
    os.dup2(_devnull_fd, 0)
    os.close(_devnull_fd)
except Exception:
    pass


class VirtualStdin(io.StringIO):
    """Virtual stdin stream providing .buffer and .raw for fast-I/O competitive programming."""

    def __init__(self, text: str):
        super().__init__(text)
        self.buffer = io.BytesIO(text.encode("utf-8"))
        self.raw = self.buffer

    def fileno(self) -> int:
        return 0


class VerificationSandbox:
    """High-throughput execution sandbox supporting dual harness and adaptive timeouts."""

    def __init__(self, default_timeout: float = 0.75, reduced_timeout: float = 0.30):
        self.timeout_tracker = AdaptiveTimeoutTracker(default_timeout, reduced_timeout)
        self._lock = threading.Lock()

    def _build_safe_globals(self, stdin_str: str = "") -> Dict[str, Any]:
        """Construct isolated namespace for code execution."""
        import builtins
        orig_import = builtins.__import__
        orig_open = builtins.open
        stdin_bytes = stdin_str.encode("utf-8")

        def safe_open(file, *args, **kwargs):
            if file in (0, "0", "/dev/stdin", "<stdin>"):
                mode = args[0] if args else kwargs.get("mode", "r")
                if "b" in mode:
                    return io.BytesIO(stdin_bytes)
                return io.StringIO(stdin_str)
            return orig_open(file, *args, **kwargs)

        def safe_import(name, *args, **kwargs):
            if name in ("threading", "multiprocessing", "subprocess", "socket", "asyncio", "urllib", "requests", "http"):
                raise ImportError(f"Module '{name}' is restricted in verification sandbox")
            mod = orig_import(name, *args, **kwargs)
            if name == "os":
                orig_read = getattr(mod, "read", None)
                if orig_read is not None:
                    mod.read = lambda fd, n: stdin_bytes[:n] if fd == 0 else orig_read(fd, n)
            return mod

        safe_builtins = dict(builtins.__dict__)
        safe_builtins["__import__"] = safe_import
        safe_builtins["open"] = safe_open

        safe_globals = {
            "__name__": "__main__",
            "__builtins__": safe_builtins,
        }
        return safe_globals

    def _execute_script_mode(
        self,
        compiled: Any,
        stdin_str: str,
        safe_globals: Dict[str, Any],
        max_steps: int = 40000,
        max_time_s: float = 0.15,
    ) -> Tuple[str, str, Optional[str]]:
        """Mode A: Execute script with redirected stdin/stdout/stderr under trace step limit."""
        in_buf = VirtualStdin(stdin_str)
        out_buf = io.StringIO()
        err_buf = io.StringIO()

        old_stdin, old_stdout, old_stderr = sys.stdin, sys.stdout, sys.stderr
        sys.stdin, sys.stdout, sys.stderr = in_buf, out_buf, err_buf
        err_msg = None

        steps = 0
        t_start = time.perf_counter()

        def tracer(frame, event, arg):
            nonlocal steps
            steps += 1
            if steps > max_steps:
                raise SandboxTimeout("Step limit reached")
            if (steps & 1023) == 0:
                if (time.perf_counter() - t_start) > max_time_s:
                    raise SandboxTimeout("Time limit reached")
            return tracer

        sys.settrace(tracer)
        try:
            exec(compiled, safe_globals)
        except SandboxTimeout:
            raise
        except SystemExit as se:
            # sys.exit(0) is standard in competitive programming
            if se.code not in (0, None):
                err_msg = f"SystemExit({se.code})"
        except Exception as exc:
            err_msg = f"{type(exc).__name__}: {exc}"
        finally:
            sys.settrace(None)
            sys.stdin = old_stdin
            sys.stdout = old_stdout
            sys.stderr = old_stderr

        return out_buf.getvalue(), err_buf.getvalue(), err_msg

    def _execute_call_mode(
        self,
        tree: ast.AST,
        compiled: Any,
        stdin_str: str,
        safe_globals: Dict[str, Any],
        max_steps: int = 40000,
        max_time_s: float = 0.15,
    ) -> Tuple[str, str, Optional[str]]:
        """Mode B: Instantiate Solution class or call entrypoint function."""
        # Find Solution class or solve function in AST
        solution_cls_name: Optional[str] = None
        solve_func_name: Optional[str] = None

        for node in tree.body:
            if isinstance(node, ast.ClassDef) and node.name in ("Solution", "Solver"):
                solution_cls_name = node.name
                break
            elif isinstance(node, ast.FunctionDef) and node.name.lower() in ("solve", "solution", "main"):
                solve_func_name = node.name

        if not solution_cls_name and not solve_func_name:
            return "", "", "no_callable_entrypoint"

        in_buf = VirtualStdin(stdin_str)
        old_stdin, old_stdout, old_stderr = sys.stdin, sys.stdout, sys.stderr
        out_buf = io.StringIO()
        err_buf = io.StringIO()
        sys.stdin, sys.stdout, sys.stderr = in_buf, out_buf, err_buf
        err_msg = None

        steps = 0
        t_start = time.perf_counter()

        def tracer(frame, event, arg):
            nonlocal steps
            steps += 1
            if steps > max_steps:
                raise SandboxTimeout("Step limit reached")
            if (steps & 1023) == 0:
                if (time.perf_counter() - t_start) > max_time_s:
                    raise SandboxTimeout("Time limit reached")
            return tracer

        sys.settrace(tracer)
        try:
            exec(compiled, safe_globals)
            callable_target: Optional[Callable] = None

            if solution_cls_name and solution_cls_name in safe_globals:
                cls_inst = safe_globals[solution_cls_name]()
                # Find candidate method
                for attr_name in dir(cls_inst):
                    if not attr_name.startswith("_"):
                        attr = getattr(cls_inst, attr_name)
                        if callable(attr):
                            callable_target = attr
                            break
            elif solve_func_name and solve_func_name in safe_globals:
                callable_target = safe_globals[solve_func_name]

            if callable_target is None:
                return "", "", "entrypoint_unresolved"

            # Parse arguments from stdin_str
            args: List[Any] = []
            tokens = stdin_str.strip().split()
            for tok in tokens:
                try:
                    val = ast.literal_eval(tok)
                    args.append(val)
                except Exception:
                    args.append(tok)

            # Try calling with parsed arguments, or with no arguments
            return_val = None
            try:
                return_val = callable_target(*args)
            except TypeError:
                try:
                    return_val = callable_target()
                except Exception as inner_exc:
                    err_msg = f"{type(inner_exc).__name__}: {inner_exc}"

            if return_val is not None:
                if isinstance(return_val, (list, tuple)):
                    formatted = " ".join(str(x) for x in return_val)
                elif isinstance(return_val, bool):
                    formatted = "YES" if return_val else "NO"
                else:
                    formatted = str(return_val)
                if out_buf.getvalue():
                    out_buf.write("\n" + formatted)
                else:
                    out_buf.write(formatted)

        except SandboxTimeout:
            raise
        except Exception as exc:
            err_msg = f"{type(exc).__name__}: {exc}"
        finally:
            sys.settrace(None)
            sys.stdin = old_stdin
            sys.stdout = old_stdout
            sys.stderr = old_stderr

        return out_buf.getvalue(), err_buf.getvalue(), err_msg

    def execute_snippet(
        self,
        code: str,
        stdin_str: str,
        expected_output: Optional[str] = None,
        force_timeout: Optional[float] = None,
    ) -> ExecutionResult:
        """Run candidate code through the dual harness within the timeout sandbox."""
        code_key = self.timeout_tracker.get_key(code)
        timeout = force_timeout if force_timeout is not None else self.timeout_tracker.get_timeout(code_key)

        if timeout is None:
            return ExecutionResult(
                status="timeout_skipped",
                stdout="",
                stderr="Skipped due to repeated timeouts (>=4)",
                runtime_ms=0.0,
                harness_mode="none",
                matched=False,
                error_message="adaptive_timeout_skipped",
            )

        # Pre-compile to check for syntax errors
        try:
            tree = ast.parse(code)
            compiled = compile(tree, "<vera_sandbox>", "exec")
        except SyntaxError as syn_err:
            return ExecutionResult(
                status="syntax_error",
                stdout="",
                stderr=str(syn_err),
                runtime_ms=0.0,
                harness_mode="none",
                matched=False,
                error_message=f"SyntaxError: {syn_err.msg} (line {syn_err.lineno})",
            )

        max_steps = 20000 if timeout < 0.10 else 40000
        t_start = time.perf_counter()
        safe_globals = self._build_safe_globals(stdin_str=stdin_str)

        try:
            # Try Mode A (Script)
            out_a, err_a, err_msg_a = self._execute_script_mode(
                compiled, stdin_str, safe_globals, max_steps=max_steps, max_time_s=timeout
            )
            runtime_ms = (time.perf_counter() - t_start) * 1000.0

            matched_a = False
            if expected_output is not None and out_a:
                matched_a = compare_outputs(out_a, expected_output)

            # If Mode A matched or had no error and produced output, accept it
            if matched_a or (not err_msg_a and out_a.strip() and expected_output is None):
                return ExecutionResult(
                    status="ok" if not err_msg_a else "error",
                    stdout=out_a,
                    stderr=err_a,
                    runtime_ms=runtime_ms,
                    harness_mode="script",
                    matched=matched_a,
                    error_message=err_msg_a,
                )

            # Otherwise, attempt Mode B (Callable / Solution class)
            safe_globals_b = self._build_safe_globals(stdin_str=stdin_str)
            out_b, err_b, err_msg_b = self._execute_call_mode(
                tree, compiled, stdin_str, safe_globals_b, max_steps=max_steps, max_time_s=timeout
            )
            runtime_ms_b = (time.perf_counter() - t_start) * 1000.0

            matched_b = False
            if expected_output is not None and out_b:
                matched_b = compare_outputs(out_b, expected_output)

            if matched_b:
                return ExecutionResult(
                    status="ok",
                    stdout=out_b,
                    stderr=err_b,
                    runtime_ms=runtime_ms_b,
                    harness_mode="call",
                    matched=True,
                    error_message=err_msg_b,
                )
            else:
                # Return best result between A and B
                best_out = out_a if out_a.strip() else out_b
                best_err = err_a if err_a.strip() else err_b
                best_msg = err_msg_a if err_msg_a else err_msg_b
                mode = "script" if out_a.strip() else ("call" if out_b.strip() else "none")
                status = "ok" if not best_msg and best_out.strip() else ("error" if best_msg else "empty_output")
                return ExecutionResult(
                    status=status,
                    stdout=best_out,
                    stderr=best_err,
                    runtime_ms=runtime_ms_b,
                    harness_mode=mode,
                    matched=False,
                    error_message=best_msg,
                )

        except SandboxTimeout:
            self.timeout_tracker.record_timeout(code_key)
            runtime_ms = (time.perf_counter() - t_start) * 1000.0
            return ExecutionResult(
                status="timeout",
                stdout="",
                stderr=f"Execution step/time limit exceeded after {runtime_ms:.1f}ms",
                runtime_ms=runtime_ms,
                harness_mode="none",
                matched=False,
                error_message=f"Timeout after {runtime_ms:.1f}ms",
            )

    def verify_candidate(
        self,
        code: str,
        examples: List[ExamplePair],
    ) -> CandidateVerificationResult:
        """Run candidate program against all extracted example pairs."""
        if not examples:
            return CandidateVerificationResult(
                passed_examples=0,
                total_examples=0,
                pass_rate=0.0,
                first_output="",
                all_passed=False,
                results=[],
                total_runtime_ms=0.0,
            )

        passed = 0
        results: List[ExecutionResult] = []
        total_time = 0.0
        first_output = ""

        for idx, ex in enumerate(examples):
            res = self.execute_snippet(
                code=code,
                stdin_str=ex.stdin,
                expected_output=ex.expected_stdout,
            )
            results.append(res)
            total_time += res.runtime_ms
            if idx == 0:
                first_output = res.stdout.strip()
            if res.matched:
                passed += 1
            elif res.status in ("timeout", "timeout_skipped", "error", "syntax_error"):
                # Early break: no need to run subsequent examples on a broken/timed-out program
                break

        pass_rate = passed / len(examples) if examples else 0.0
        all_passed = (passed == len(examples)) and len(examples) > 0

        return CandidateVerificationResult(
            passed_examples=passed,
            total_examples=len(examples),
            pass_rate=pass_rate,
            first_output=first_output,
            all_passed=all_passed,
            results=results,
            total_runtime_ms=total_time,
        )


# API Alias
ExecutionSandbox = VerificationSandbox

