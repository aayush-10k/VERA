"""Unit tests for WorkedExampleParser and static AST SignatureExtractor."""

import pytest
from vera.verify.parser import WorkedExampleParser, ExamplePair
from vera.gate.signature import extract_program_signature, ProgramSignature
from vera.data.loader import AppsRetrievalDataset


SAMPLE_CF_STATEMENT = """
A. Lucky Numbers
time limit per test: 1.0 s
memory limit per test: 256 MB

Olympian gods love lucky numbers. An integer is lucky if it contains only digits 4 and 7.
Find the n-th lucky number.

-----Sample Input-----
2
-----Sample Output-----
7

-----Sample Input-----
3
-----Sample Output-----
44
"""

SAMPLE_LEETCODE_STATEMENT = """
Given an array of integers nums and an integer target, return indices of the two numbers.

Example 1:
Input: nums = [2,7,11,15], target = 9
Output: [0,1]

Example 2:
Input: nums = [3,2,4], target = 6
Output: [1,2]
"""


def test_parser_codeforces_format():
    parser = WorkedExampleParser()
    examples = parser.parse_examples(SAMPLE_CF_STATEMENT)

    assert len(examples) == 2
    assert examples[0].stdin.strip() == "2"
    assert examples[0].expected_stdout.strip() == "7"
    assert examples[1].stdin.strip() == "3"
    assert examples[1].expected_stdout.strip() == "44"


def test_parser_leetcode_format():
    parser = WorkedExampleParser()
    examples = parser.parse_examples(SAMPLE_LEETCODE_STATEMENT)

    assert len(examples) >= 2
    assert "nums = [2,7,11,15]" in examples[0].stdin
    assert "[0,1]" in examples[0].expected_stdout


def test_parser_coverage_on_dataset():
    ds = AppsRetrievalDataset()
    test_queries = ds.get_test_queries()
    parser = WorkedExampleParser()

    # Sample 100 actual APPS test queries
    sample_qids = list(test_queries.keys())[:100]
    parsed_count = 0

    for qid in sample_qids:
        ex = parser.parse_examples(test_queries[qid])
        if len(ex) > 0:
            parsed_count += 1

    # Verify acceptance criteria: parse rate >= 75%
    parse_rate = (parsed_count / len(sample_qids)) * 100
    assert parse_rate >= 75.0, f"Parse rate {parse_rate:.1f}% below 75% threshold"


def test_signature_extractor_shapes():
    # 1. Single line
    code_single = "x = int(input())\nprint(x * 2)"
    sig_single = extract_program_signature(code_single)
    assert sig_single.io_shape in ("single-line", "token-line")
    assert sig_single.input_calls == 1

    # 2. Multi-case loop
    code_multi = """
t = int(input())
for _ in range(t):
    n = int(input())
    print(n + 1)
"""
    sig_multi = extract_program_signature(code_multi)
    assert sig_multi.io_shape == "multi-case-t"
    assert sig_multi.has_outer_loop is True

    # 3. Functional class
    code_func = """
class Solution:
    def solve(self, nums):
        return sorted(nums)
"""
    sig_func = extract_program_signature(code_func)
    assert sig_func.io_shape == "functional"
    assert sig_func.has_solution_class is True
    assert sig_func.input_calls == 0
