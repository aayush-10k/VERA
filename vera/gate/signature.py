"""
VERA Static AST Signature Extractor
===================================
Analyzes candidate Python programs via static AST parsing:
1. Classifies I/O shape: {single-line, n-then-n-lines, token-line, multi-case-t, unknown}
2. Identifies functional entrypoints (Solution.solve / solve()) vs stdin scripts
3. Counts input() / sys.stdin invocations to guard against hanging programs
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from typing import List, Optional, Set


@dataclass(frozen=True)
class ProgramSignature:
    io_shape: str  # 'single-line', 'n-then-n-lines', 'token-line', 'multi-case-t', 'functional', 'unknown'
    input_calls: int
    has_solve_func: bool
    has_solution_class: bool
    uses_stdin: bool
    uses_split: bool
    has_outer_loop: bool
    is_valid_syntax: bool
    syntax_error: Optional[str] = None


class SignatureExtractor(ast.NodeVisitor):
    """AST visitor extracting structural I/O and signature traits."""

    def __init__(self):
        self.input_calls = 0
        self.has_solve_func = False
        self.has_solution_class = False
        self.uses_stdin = False
        self.uses_split = False
        self.has_outer_loop = False
        self.has_testcase_loop = False
        self.has_n_loop = False

    def visit_Call(self, node: ast.Call):
        # Detect input()
        if isinstance(node.func, ast.Name) and node.func.id == "input":
            self.input_calls += 1
        # Detect sys.stdin.readline() / sys.stdin.read()
        elif isinstance(node.func, ast.Attribute):
            if node.func.attr in ("readline", "read", "readlines"):
                self.uses_stdin = True
                self.input_calls += 1
            elif node.func.attr == "split":
                self.uses_split = True
        self.generic_visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef):
        if node.name.lower() in ("solve", "solution", "main"):
            self.has_solve_func = True
        self.generic_visit(node)

    def visit_ClassDef(self, node: ast.ClassDef):
        if node.name.lower() in ("solution", "solver"):
            self.has_solution_class = True
        self.generic_visit(node)

    def visit_For(self, node: ast.For):
        self.has_outer_loop = True
        # Check for `for _ in range(t):` or `for _ in range(int(input())):`
        if isinstance(node.iter, ast.Call):
            if isinstance(node.iter.func, ast.Name) and node.iter.func.id == "range":
                if node.iter.args:
                    arg0 = node.iter.args[0]
                    # Check if argument is int(input()) or variable
                    if isinstance(arg0, ast.Call) and isinstance(arg0.func, ast.Name) and arg0.func.id == "int":
                        self.has_testcase_loop = True
                    elif isinstance(arg0, ast.Name) and arg0.id.lower() in ("t", "test", "tc"):
                        self.has_testcase_loop = True
                    elif isinstance(arg0, ast.Name) and arg0.id.lower() in ("n", "m", "k"):
                        self.has_n_loop = True
        self.generic_visit(node)

    def visit_While(self, node: ast.While):
        self.has_outer_loop = True
        self.generic_visit(node)


def extract_program_signature(code: str) -> ProgramSignature:
    """
    Parses Python code and extracts its structural ProgramSignature.
    """
    if not code or not isinstance(code, str):
        return ProgramSignature(
            io_shape="unknown",
            input_calls=0,
            has_solve_func=False,
            has_solution_class=False,
            uses_stdin=False,
            uses_split=False,
            has_outer_loop=False,
            is_valid_syntax=False,
            syntax_error="Empty or non-string input",
        )

    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return ProgramSignature(
            io_shape="unknown",
            input_calls=0,
            has_solve_func=False,
            has_solution_class=False,
            uses_stdin=False,
            uses_split=False,
            has_outer_loop=False,
            is_valid_syntax=False,
            syntax_error=str(e),
        )

    visitor = SignatureExtractor()
    visitor.visit(tree)

    # Classify I/O shape
    if visitor.has_solution_class or (visitor.has_solve_func and visitor.input_calls == 0):
        io_shape = "functional"
    elif visitor.has_testcase_loop:
        io_shape = "multi-case-t"
    elif visitor.has_n_loop:
        io_shape = "n-then-n-lines"
    elif visitor.uses_split:
        io_shape = "token-line"
    elif visitor.input_calls == 1:
        io_shape = "single-line"
    elif visitor.input_calls > 1:
        io_shape = "token-line"
    else:
        io_shape = "unknown"

    return ProgramSignature(
        io_shape=io_shape,
        input_calls=visitor.input_calls,
        has_solve_func=visitor.has_solve_func,
        has_solution_class=visitor.has_solution_class,
        uses_stdin=visitor.uses_stdin,
        uses_split=visitor.uses_split,
        has_outer_loop=visitor.has_outer_loop,
        is_valid_syntax=True,
    )
