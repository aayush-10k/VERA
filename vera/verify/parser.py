"""
VERA Worked-Example Parser
==========================
Extracts structured ``(stdin, expected_stdout)`` test fixtures from APPS problem
statements. The statement formats present in CoIR AppsRetrieval (measured on the
3,765 test queries in ``docs/dataset-audit.md``) are, in order of frequency:

1. **Codeforces** (~79%)::

       -----Examples-----
       Input
       3
       1 2 3

       Output
       6

       Input
       ...
       -----Note-----

2. **AtCoder / CodeChef dashed samples** (~19%)::

       -----Sample Input-----        -----Sample Input:-----      -----Example Input-----
       abaababaab                    5                            bbccdd
       -----Sample Output-----       -----Sample Output:-----     -----Example Output-----
       6                             0 1                          1

       <free-text explanation follows after a blank line>

3. **LeetCode** (<1% of test, ~12% of the train partition)::

       Example 1:
       Input: nums = [2,7,11,15], target = 9
       Output: [0,1]
       Explanation: ...

4. **Bare markers** (rare): ``Sample Input 1:`` / ``Sample Output 1:`` lines without dashes.

The specification sections ``-----Input-----`` / ``-----Output-----`` describe the
format in prose and are *never* examples; the previous heuristic parser confused
them with examples, which is what produced the spurious 6% gold-run rate.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Optional, Tuple


@dataclass(frozen=True)
class ExamplePair:
    stdin: str
    expected_stdout: str
    example_index: int
    source_format: str


@dataclass
class ParseReport:
    examples: List[ExamplePair] = field(default_factory=list)
    source_format: str = "none"
    reason: str = ""  # '' when examples were found, else a failure category


# ---- regexes ---------------------------------------------------------------- #
# A dashed section header such as "-----Examples-----" or "-----Sample Input:-----"
_DASHED_HEADER_RE = re.compile(r"^[ \t]*-{3,}[ \t]*(?P<name>[^\n-][^\n]*?)?[ \t]*-{3,}[ \t]*$", re.MULTILINE)
_EXAMPLES_NAME_RE = re.compile(r"^examples?\s*:?$", re.IGNORECASE)
_SAMPLE_IN_NAME_RE = re.compile(r"^(?:sample|example|test)\s*input\s*(?P<num>\d*)\s*:?$", re.IGNORECASE)
_SAMPLE_OUT_NAME_RE = re.compile(r"^(?:sample|example|test)\s*output\s*(?P<num>\d*)\s*:?$", re.IGNORECASE)

# Marker lines *inside* a Codeforces examples section
_IO_MARKER_RE = re.compile(r"^[ \t]*(?P<kind>Input|Output)[ \t]*:?[ \t]*$", re.IGNORECASE | re.MULTILINE)
# Bare (non-dashed) sample markers anywhere in the statement
_BARE_MARKER_RE = re.compile(
    r"^[ \t]*(?:(?:Sample|Example|Test)[ \t]*)?(?P<kind>Input|Output)[ \t]*(?:#?\s*\d+)?[ \t]*:?[ \t]*$",
    re.IGNORECASE | re.MULTILINE,
)

# LeetCode "Example 1:\nInput: ...\nOutput: ..."
_LEETCODE_RE = re.compile(
    r"Example\s*\d*\s*:?[ \t]*\n\s*Input\s*:?[ \t]*(?P<inp>.*?)\n\s*Output\s*:?[ \t]*(?P<out>.*?)"
    r"(?=\n[ \t]*(?:Explanation|Explaination|Example|Note|Constraints|Follow|Hint|Output|Input)\b|\n[ \t]*\n|\Z)",
    re.DOTALL | re.IGNORECASE,
)
_INPUT_OUTPUT_RE = re.compile(
    r"^[ \t]*Input\s*:[ \t]*(?P<inp>.*?)\n\s*Output\s*:[ \t]*(?P<out>.*?)"
    r"(?=\n[ \t]*(?:Explanation|Explaination|Example|Note|Constraints|Follow|Hint|Output|Input)\b|\n[ \t]*\n|\Z)",
    re.DOTALL | re.IGNORECASE | re.MULTILINE,
)

# Prose detector: the format-specification text that must never be mistaken for data.
_PROSE_HINTS = re.compile(
    r"\b(the first line|first line|second line|each line|contains?|denot|consist|integer[s]?\b.*\b(?:and|or)\b|"
    r"description|test case[s]? follow|separated by|following format|given from standard input)\b",
    re.IGNORECASE,
)


def _normalize_block(text: str) -> str:
    """Normalize newlines / nbsp, strip trailing spaces per line and surrounding blank lines; ensure one trailing \\n."""
    t = text.replace("\r\n", "\n").replace("\r", "\n").replace("\xa0", " ")
    t = _strip_latex_dollars(t)
    lines = [ln.rstrip() for ln in t.split("\n")]
    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop()
    if not lines:
        return ""
    return "\n".join(lines) + "\n"


def _is_data_like(paragraph: str) -> bool:
    """True for sample-output-looking text (numbers, short tokens, verdict words), False for explanations."""
    par = paragraph.strip()
    if not par:
        return False
    if par.startswith(("- ", "* ", "\u2022")) or _looks_like_prose(par):
        return False
    tokens = par.split()
    long_words = [t for t in tokens if re.fullmatch(r"[A-Za-z][A-Za-z']{3,}[.,;:!?]?", t)]
    if len(tokens) <= 3:
        return True
    return len(long_words) <= max(2, int(0.2 * len(tokens))) and not re.search(r"[.!?]\s*$", par)


def _first_paragraph(text: str) -> str:
    """Keep the leading data-like paragraphs of an output block; drop the free-text explanation that AtCoder/CodeChef
    statements append after a blank line (multi-case outputs separated by blank lines are kept)."""
    t = text.replace("\r\n", "\n").replace("\r", "\n")
    t = re.sub(r"^\s*\n", "", t)
    parts = re.split(r"\n[ \t]*\n", t)
    keep = []
    for i, par in enumerate(parts):
        if i == 0 or _is_data_like(par):
            keep.append(par)
        else:
            break
    return "\n\n".join(keep)


_LATEX_TOKEN_RE = re.compile(r"\$([^$\n]{1,40})\$")


def _strip_latex_dollars(block: str) -> str:
    """CodeChef statements wrap sample numbers in $...$; unwrap when every $...$ group is a plain token."""
    if "$" not in block:
        return block
    groups = _LATEX_TOKEN_RE.findall(block)
    if groups and all(re.fullmatch(r"[-+]?[\w.,]+", g.strip()) for g in groups):
        block = _LATEX_TOKEN_RE.sub(lambda m: m.group(1).strip(), block)
    return block


def _looks_like_prose(block: str) -> bool:
    b = block.strip()
    if len(b) < 30:
        return False
    words = re.findall(r"[A-Za-z]{3,}", b)
    if len(words) >= 6 and _PROSE_HINTS.search(b):
        return True
    # mostly long alphabetic words and sentence punctuation -> prose
    alpha_ratio = len("".join(words)) / max(1, len(b))
    return alpha_ratio > 0.55 and len(words) >= 10 and ("." in b or "," in b)


class WorkedExampleParser:
    """Parses executable input/output test pairs from problem statements."""

    def parse_examples(self, statement: str) -> List[ExamplePair]:
        """Extract all worked example pairs from a problem statement (empty list if none)."""
        return self.parse_report(statement).examples

    parse = parse_examples

    # ------------------------------------------------------------------ #
    def parse_report(self, statement: str) -> ParseReport:
        if not statement or not isinstance(statement, str):
            return ParseReport(reason="empty_statement")
        text = statement.replace("\r\n", "\n").replace("\r", "\n")

        for fmt, fn in (
            ("codeforces", self._parse_codeforces),
            ("dashed_sample", self._parse_dashed_samples),
            ("leetcode", self._parse_leetcode),
            ("bare_markers", self._parse_bare_markers),
        ):
            pairs, reason = fn(text)
            if pairs:
                return ParseReport(
                    examples=[ExamplePair(s, o, i, fmt) for i, (s, o) in enumerate(pairs)],
                    source_format=fmt,
                    reason="",
                )
            if reason and reason not in ("no_section",):
                # A format matched structurally but yielded nothing usable; keep looking but remember why.
                last_reason = reason
            else:
                last_reason = None
            if last_reason:
                remembered = last_reason
        # Nothing worked: categorize.
        if "[Image]" in text and not re.search(r"(?i)example|sample", text):
            return ParseReport(reason="image_only_or_no_example_section")
        if not re.search(r"(?i)\b(example|sample)\b", text):
            return ParseReport(reason="no_example_section")
        try:
            return ParseReport(reason=remembered)  # type: ignore[name-defined]
        except NameError:
            return ParseReport(reason="example_section_unparsed")

    # ------------------------------------------------------------------ #
    @staticmethod
    def _headers(text: str) -> List[Tuple[int, int, str]]:
        """All dashed headers as (start, end, name)."""
        out = []
        for m in _DASHED_HEADER_RE.finditer(text):
            name = (m.group("name") or "").strip()
            out.append((m.start(), m.end(), name))
        return out

    def _parse_codeforces(self, text: str) -> Tuple[List[Tuple[str, str]], str]:
        headers = self._headers(text)
        sections = [(i, h) for i, h in enumerate(headers) if _EXAMPLES_NAME_RE.match(h[2])]
        if not sections:
            return [], "no_section"
        pairs: List[Tuple[str, str]] = []
        for i, (_s, end, _name) in sections:
            body_end = headers[i + 1][0] if i + 1 < len(headers) else len(text)
            body = text[end:body_end]
            pairs.extend(self._pairs_from_markers(body, _IO_MARKER_RE))
        pairs = [(s, o) for s, o in pairs if s.strip() and o.strip() and not _looks_like_prose(s)]
        return pairs, ("" if pairs else "examples_section_without_io_blocks")

    @staticmethod
    def _pairs_from_markers(body: str, marker_re: "re.Pattern[str]") -> List[Tuple[str, str]]:
        markers = [(m.start(), m.end(), m.group("kind").lower()) for m in marker_re.finditer(body)]
        pairs: List[Tuple[str, str]] = []
        i = 0
        while i < len(markers):
            if markers[i][2] != "input":
                i += 1
                continue
            # find the next output marker
            j = i + 1
            while j < len(markers) and markers[j][2] != "output":
                j += 1
            if j >= len(markers):
                break
            stdin_block = body[markers[i][1]: markers[j][0]]
            out_end = markers[j + 1][0] if j + 1 < len(markers) else len(body)
            expected_block = body[markers[j][1]: out_end]
            pairs.append((_normalize_block(stdin_block), _normalize_block(expected_block)))
            i = j + 1
        return pairs

    def _parse_dashed_samples(self, text: str) -> Tuple[List[Tuple[str, str]], str]:
        headers = self._headers(text)
        pairs: List[Tuple[str, str]] = []
        seen_input = False
        for idx, (_s, end, name) in enumerate(headers):
            if not _SAMPLE_IN_NAME_RE.match(name):
                continue
            seen_input = True
            if idx + 1 >= len(headers) or not _SAMPLE_OUT_NAME_RE.match(headers[idx + 1][2]):
                continue
            stdin_block = text[end: headers[idx + 1][0]]
            out_start = headers[idx + 1][1]
            out_end = headers[idx + 2][0] if idx + 2 < len(headers) else len(text)
            expected_block = _first_paragraph(text[out_start:out_end])
            s, o = _normalize_block(stdin_block), _normalize_block(expected_block)
            if s.strip() and o.strip() and not _looks_like_prose(s):
                pairs.append((s, o))
        if pairs:
            return pairs, ""
        return [], ("sample_headers_unpaired" if seen_input else "no_section")

    def _parse_leetcode(self, text: str) -> Tuple[List[Tuple[str, str]], str]:
        pairs: List[Tuple[str, str]] = []
        matches = list(_LEETCODE_RE.finditer(text)) or list(_INPUT_OUTPUT_RE.finditer(text))
        if not matches:
            return [], "no_section"
        for m in matches:
            s = _normalize_block(m.group("inp"))
            o = _normalize_block(_first_paragraph(m.group("out")))
            if s.strip() and o.strip() and not _looks_like_prose(s) and not _looks_like_prose(o):
                pairs.append((s, o))
        return pairs, ("" if pairs else "leetcode_blocks_unusable")

    def _parse_bare_markers(self, text: str) -> Tuple[List[Tuple[str, str]], str]:
        # Remove dashed header lines first so "-----Input-----" spec headers cannot act as markers.
        stripped = _DASHED_HEADER_RE.sub("", text)
        # Only consider markers located after the first mention of Example/Sample to skip the spec section.
        m = re.search(r"(?i)\b(examples?|samples?)\b", stripped)
        if not m:
            return [], "no_section"
        region = stripped[m.start():]
        pairs = self._pairs_from_markers(region, _BARE_MARKER_RE)
        pairs = [(s, o) for s, o in pairs if s.strip() and o.strip() and not _looks_like_prose(s) and not _looks_like_prose(o)]
        return pairs, ("" if pairs else "bare_markers_unusable")


def statement_allows_any_order(statement: str) -> bool:
    """True when the statement says outputs may be printed in any order (enables set comparison)."""
    return bool(re.search(r"(?i)\b(in any order|any order|any valid|any of them|print any)\b", statement or ""))
