"""
VERA Chassis Preprocessing
==========================
Preprocesses problem statements and solution corpus:
1. Strips competitive programming boilerplate (fast-IO aliases, recursion limit overrides,
   threading banners) from Python solutions to maximize dense semantic signal.
2. Normalizes problem queries up to 8192 tokens to capture tail worked examples.
3. Retains both raw and stripped representations (raw for verification, stripped for dense encoding).
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple


# Regex patterns for Python competitive programming boilerplate
BOILERPLATE_PATTERNS = [
    # Fast I/O rebinding
    re.compile(r"^\s*(?:import\s+sys|from\s+sys\s+import\s+stdin[^\n]*)", re.MULTILINE),
    re.compile(r"^\s*input\s*=\s*(?:sys\.)?stdin\.readline[^\n]*", re.MULTILINE),
    re.compile(r"^\s*print\s*=\s*(?:sys\.)?stdout\.write[^\n]*", re.MULTILINE),
    # Recursion limit overrides
    re.compile(r"^\s*(?:sys\.)?setrecursionlimit\([^\)]*\)", re.MULTILINE),
    re.compile(r"^\s*threading\.stack_size\([^\)]*\)", re.MULTILINE),
    # Generic contest macro headers
    re.compile(r"^\s*INF\s*=\s*(?:float\(['\"]inf['\"]\)|\d+e\d+|\d{9,})", re.MULTILINE),
    re.compile(r"^\s*MOD\s*=\s*(?:10\*\*9\s*\+\s*7|998244353)", re.MULTILINE),
    # Repetitive template comments
    re.compile(r"^\s*#\s*-*.*(?:template|author|created|solution).*", re.IGNORECASE | re.MULTILINE),
]


def strip_corpus_boilerplate(code: str) -> str:
    """
    Strips template and fast-IO boilerplate from candidate code snippets.
    Leaves algorithmic logic intact.
    """
    if not code or not isinstance(code, str):
        return ""

    cleaned = code
    for pat in BOILERPLATE_PATTERNS:
        cleaned = pat.sub("", cleaned)

    # Collapse excessive blank lines
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def preprocess_query(
    text: str,
    max_chars: int = 32000,
    remove_examples: bool = False,
) -> str:
    """
    Normalizes query problem statements up to 8192 tokens (~32,000 characters).
    Optional: remove worked example section from dense embedding if testing variant B.
    """
    if not text or not isinstance(text, str):
        return ""

    normalized = text.strip()

    if remove_examples:
        # Strip out explicit example sections for A/B testing
        split_match = re.split(r"(?i)\n(?:examples?|sample\s+input)", normalized, maxsplit=1)
        if len(split_match) > 1 and len(split_match[0].strip()) > 100:
            normalized = split_match[0].strip()

    # Normalize excessive whitespace
    normalized = re.sub(r"[ \t]+", " ", normalized)
    normalized = re.sub(r"\n{3,}", "\n\n", normalized)

    # 8192 context budget cap
    if len(normalized) > max_chars:
        normalized = normalized[:max_chars]

    return normalized


class ChassisPreprocessor:
    """Manages dual corpus representations (raw for verify, stripped for dense)."""

    def __init__(self):
        self.stripped_corpus: Dict[str, str] = {}
        self.raw_corpus: Dict[str, str] = {}

    def process_corpus(self, corpus: Dict[str, str]) -> Dict[str, str]:
        """Processes and caches stripped corpus while preserving raw."""
        self.raw_corpus = corpus
        self.stripped_corpus = {
            doc_id: strip_corpus_boilerplate(code)
            for doc_id, code in corpus.items()
        }
        return self.stripped_corpus

    def process_queries(
        self,
        queries: Dict[str, str],
        remove_examples: bool = False,
    ) -> Dict[str, str]:
        """Preprocesses query dictionary."""
        return {
            qid: preprocess_query(text, remove_examples=remove_examples)
            for qid, text in queries.items()
        }
