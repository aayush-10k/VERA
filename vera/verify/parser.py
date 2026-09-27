"""
VERA Worked-Example Parser
==========================
Extracts structured (stdin, expected_stdout) test fixtures from
natural language problem statements in the APPS dataset.
Handles multiple example formats, code fences, and multi-line I/O.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional, Tuple


@dataclass(frozen=True)
class ExamplePair:
    stdin: str
    expected_stdout: str
    example_index: int
    source_format: str


class WorkedExampleParser:
    """Parses executable input/output test pairs from problem statements."""

    # Regex patterns for different APPS problem statement formats
    PATTERNS = [
        # Format 1: Codeforces style with optional leading/trailing dashes: -----Sample Input-----
        (
            "codeforces_sample_io",
            re.compile(
                r"(?:-+\s*)?(?:Sample\s+Input|Input\s*\d*):?(?:\s*-+)?\s*\n```?(.*?)```?\s*(?:-+\s*)?(?:Sample\s+Output|Output\s*\d*):?(?:\s*-+)?\s*\n```?(.*?)```?(?=\n(?:-+)?\s*Sample\s+Input|\n(?:-+)?\s*Note|\n(?:-+)?\s*Example|\Z)",
                re.DOTALL | re.IGNORECASE,
            ),
        ),
        # Format 2: Standard LeetCode style "Example 1:\nInput: ...\nOutput: ..."
        (
            "leetcode_example_io",
            re.compile(
                r"(?:Example\s*\d*:?|Sample\s*\d*:?)\s*\n\s*(?:Input|stdin)\s*:\s*(.*?)\n\s*(?:Output|stdout)\s*:\s*(.*?)(?=\n\s*(?:Example|\Z|Explanation))",
                re.DOTALL | re.IGNORECASE,
            ),
        ),
        # Format 3: Markdown code blocks under Example heading
        (
            "fenced_example_io",
            re.compile(
                r"(?:###?\s*Examples?|Examples?)\s*:\s*\n```(?:[a-z]*\n)?(.*?)```\s*\n```(?:[a-z]*\n)?(.*?)```",
                re.DOTALL | re.IGNORECASE,
            ),
        ),
        # Format 4: Generic labeled blocks
        (
            "generic_io_blocks",
            re.compile(
                r"(?:Input|Sample\s*Input)\s*\n(.*?)\n(?:Output|Sample\s*Output)\s*\n(.*?)(?=\n(?:Input|Sample|Note|\Z))",
                re.DOTALL | re.IGNORECASE,
            ),
        ),
    ]

    def parse_examples(self, statement: str) -> List[ExamplePair]:
        """
        Extracts all worked example pairs from a problem statement.
        Returns list of ExamplePair dataclasses.
        """
        if not statement or not isinstance(statement, str):
            return []

        examples: List[ExamplePair] = []

        # Try regex patterns in order of specificity
        for format_name, pat in self.PATTERNS:
            matches = list(pat.finditer(statement))
            if matches:
                for idx, m in enumerate(matches):
                    stdin_text = self._clean_io_text(m.group(1))
                    stdout_text = self._clean_io_text(m.group(2))

                    if stdin_text is not None and stdout_text is not None:
                        examples.append(
                            ExamplePair(
                                stdin=stdin_text,
                                expected_stdout=stdout_text,
                                example_index=idx,
                                source_format=format_name,
                            )
                        )
                if examples:
                    return examples

        # Fallback: line-by-line heuristic parser for unstructured statements
        fallback_examples = self._parse_line_by_line(statement)
        if fallback_examples:
            return fallback_examples

        return []

    def _clean_io_text(self, text: str) -> str:
        """Strips markdown fences, surrounding quotes, and normalizes linebreaks."""
        t = text.strip()
        # Remove markdown fences
        t = re.sub(r"^```[a-zA-Z0-9_-]*\n", "", t)
        t = re.sub(r"\n```$", "", t)
        # Normalize newlines
        t = t.replace("\r\n", "\n").replace("\r", "\n")
        return t.strip() + "\n"

    def _parse_line_by_line(self, text: str) -> List[ExamplePair]:
        """Heuristic line-based parser when standard regex delimiters miss."""
        lines = text.splitlines()
        input_lines: List[str] = []
        output_lines: List[str] = []
        state = None
        examples = []

        for line in lines:
            stripped = line.strip()
            if re.match(r"(?i)^(?:-+\s*)?(?:sample\s+)?input\s*[:\d]*(?:\s*-+)?$", stripped):
                if input_lines and output_lines:
                    examples.append(
                        ExamplePair(
                            stdin="\n".join(input_lines).strip() + "\n",
                            expected_stdout="\n".join(output_lines).strip() + "\n",
                            example_index=len(examples),
                            source_format="heuristic_lines",
                        )
                    )
                    input_lines, output_lines = [], []
                state = "input"
                continue
            elif re.match(r"(?i)^(?:-+\s*)?(?:sample\s+)?output\s*[:\d]*(?:\s*-+)?$", stripped):
                state = "output"
                continue
            elif re.match(r"(?i)^(?:note|explanation|example\s*\d*)$", stripped):
                state = None
                continue

            if state == "input":
                input_lines.append(line)
            elif state == "output":
                output_lines.append(line)

        if input_lines and output_lines:
            examples.append(
                ExamplePair(
                    stdin="\n".join(input_lines).strip() + "\n",
                    expected_stdout="\n".join(output_lines).strip() + "\n",
                    example_index=len(examples),
                    source_format="heuristic_lines",
                )
            )

        return examples

    parse = parse_examples
