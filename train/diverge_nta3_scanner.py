"""Source-only full-document transaction scanner for DIVERGE-NTA3."""

from __future__ import annotations

import re


class NTA3ScannerError(RuntimeError):
    """A context-rich source document violates the frozen section grammar."""


START = "Candidate reasoning:\n"
END = "\n\nCandidate final answer:"
TRANSACTION = re.compile(r"-?\d+\s*[+\-*]\s*\d+\s*=\s*-?\d+")


def scan_transactions(document: str) -> tuple[str, ...]:
    if document.count(START) != 1 or document.count(END) != 1:
        raise NTA3ScannerError("natural document section markers differ")
    start = document.index(START) + len(START)
    end = document.index(END, start)
    if end <= start:
        raise NTA3ScannerError("natural reasoning section is empty")
    source = document[start:end]
    matches = tuple(match.group(0) for match in TRANSACTION.finditer(source))
    if not 2 <= len(matches) <= 5:
        raise NTA3ScannerError("natural transaction count differs")
    return matches


__all__ = ["NTA3ScannerError", "scan_transactions"]
