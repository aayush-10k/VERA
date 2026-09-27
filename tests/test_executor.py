"""
Tests for VERA Verification Engine Sandbox & Normalized Output Comparator
=========================================================================
"""

import time
import pytest
from vera.verify.comparator import compare_outputs, evaluate_comparison
from vera.verify.executor import ExecutionSandbox, AdaptiveTimeoutTracker
from vera.verify.parser import ExamplePair


class TestComparator:
    """Test normalized output comparator against competitive programming output variations."""

    def test_exact_match(self):
        assert compare_outputs("42\n", "42")
        assert compare_outputs("hello world\n\n", "hello world")

    def test_whitespace_and_newlines(self):
        assert compare_outputs("1 2 3\n", "1\n2\n3")
        assert compare_outputs("  a   b   c  \n", "a b c")

    def test_float_tolerance(self):
        # 1e-7 difference should pass 1e-6 tolerance
        assert compare_outputs("3.14159265", "3.14159260", float_tol=1e-5)
        # Large difference should fail
        assert not compare_outputs("3.14", "3.24", float_tol=1e-5)
        # Mixed floats and strings
        assert compare_outputs("result: 1.0000001", "result: 1.0000002", float_tol=1e-5)

    def test_case_insensitive_verdicts(self):
        assert compare_outputs("YES\n", "Yes")
        assert compare_outputs("no", "NO")
        assert compare_outputs("True\n", "true")
        assert compare_outputs("possible", "POSSIBLE")

    def test_empty_outputs(self):
        assert compare_outputs("", "")
        assert compare_outputs("   \n", "")
        assert not compare_outputs("42", "")
        assert not compare_outputs("", "42")

    def test_multiline_permutation(self):
        actual = "apple\nbanana\ncherry"
        expected = "cherry\napple\nbanana"
        assert compare_outputs(actual, expected, multiline_set=True)


class TestVerificationSandbox:
    """Test sandbox execution, dual harness, and timeout behavior."""

    @pytest.fixture
    def sandbox(self):
        return ExecutionSandbox(default_timeout=0.5, reduced_timeout=0.2)

    def test_mode_a_script_execution(self, sandbox):
        code = "n = int(input())\nprint(n * 2)"
        res = sandbox.execute_snippet(code, stdin_str="21\n", expected_output="42")
        assert res.status == "ok"
        assert res.matched is True
        assert res.stdout.strip() == "42"
        assert res.harness_mode == "script"

    def test_mode_b_solution_class(self, sandbox):
        code = """
class Solution:
    def solve(self, a, b):
        return a + b
"""
        res = sandbox.execute_snippet(code, stdin_str="15 25", expected_output="40")
        assert res.status == "ok"
        assert res.matched is True
        assert res.stdout.strip() == "40"
        assert res.harness_mode == "call"

    def test_sys_exit_zero_graceful(self, sandbox):
        code = "import sys\nprint('DONE')\nsys.exit(0)"
        res = sandbox.execute_snippet(code, stdin_str="", expected_output="DONE")
        assert res.status == "ok"
        assert res.matched is True
        assert res.stdout.strip() == "DONE"

    def test_syntax_error_detection(self, sandbox):
        code = "def bad_syntax(:\n    pass"
        res = sandbox.execute_snippet(code, stdin_str="")
        assert res.status == "syntax_error"
        assert res.matched is False

    def test_timeout_and_adaptive_demotion(self, sandbox):
        code = "while True:\n    pass"
        code_key = sandbox.timeout_tracker.get_key(code)

        # 1st run: times out with default timeout (0.5s)
        res1 = sandbox.execute_snippet(code, stdin_str="", force_timeout=0.2)
        assert res1.status == "timeout"
        assert sandbox.timeout_tracker._counts[code_key] == 1

        # 2nd run
        res2 = sandbox.execute_snippet(code, stdin_str="", force_timeout=0.1)
        assert res2.status == "timeout"
        assert sandbox.timeout_tracker._counts[code_key] == 2

        # 3rd run
        res3 = sandbox.execute_snippet(code, stdin_str="", force_timeout=0.1)
        assert sandbox.timeout_tracker._counts[code_key] == 3

        # 4th run
        res4 = sandbox.execute_snippet(code, stdin_str="", force_timeout=0.1)
        assert sandbox.timeout_tracker._counts[code_key] == 4

        # 5th run: should be skipped automatically without waiting!
        t0 = time.perf_counter()
        res5 = sandbox.execute_snippet(code, stdin_str="")
        t1 = time.perf_counter()
        assert res5.status == "timeout_skipped"
        assert (t1 - t0) < 0.05  # Skipped instantly

    def test_verify_candidate_multiple_examples(self, sandbox):
        code = """
import sys
line = sys.stdin.read().strip()
if line:
    print(int(line) ** 2)
"""
        examples = [
            ExamplePair(stdin="3\n", expected_stdout="9\n", example_index=0, source_format="test"),
            ExamplePair(stdin="4\n", expected_stdout="16\n", example_index=1, source_format="test"),
            ExamplePair(stdin="5\n", expected_stdout="25\n", example_index=2, source_format="test"),
        ]
        cand_res = sandbox.verify_candidate(code, examples)
        assert cand_res.all_passed is True
        assert cand_res.passed_examples == 3
        assert cand_res.pass_rate == 1.0
        assert cand_res.first_output == "9"

    def test_execution_throughput(self, sandbox):
        code = "x = int(input())\nprint(x + 1)"
        t0 = time.perf_counter()
        for i in range(100):
            res = sandbox.execute_snippet(code, stdin_str=f"{i}\n", expected_output=f"{i + 1}")
            assert res.matched is True
        t1 = time.perf_counter()
        total_ms = (t1 - t0) * 1000
        # 100 runs on a warm sandbox
        # fork-per-run, no interpreter start-up: well under 20 ms/run even on a loaded 4-core box
        assert total_ms < 2000, f"Took {total_ms:.2f} ms"

    def test_open_0_fast_io(self, sandbox):
        code = "a, b = map(int, open(0).read().split())\nprint(a * b)"
        res = sandbox.execute_snippet(code, stdin_str="6 7\n", expected_output="42")
        assert res.matched is True
        assert res.stdout.strip() == "42"

    def test_os_read_fast_io(self, sandbox):
        code = "import os\ndata = os.read(0, 1024).decode('utf-8').strip()\nprint(data.upper())"
        res = sandbox.execute_snippet(code, stdin_str="hello\n", expected_output="HELLO")
        assert res.matched is True
        assert res.stdout.strip() == "HELLO"

