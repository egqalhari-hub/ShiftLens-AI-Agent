"""Verifier agent — the critic (DESIGN.md v2).

Extracts every numeric claim from a draft (ints, decimals, percents — via
regex) and re-checks each against the numbers the tools actually produced.
A claim passes if it matches some tool-computed number, allowing for the
rounding a drafter may legitimately have applied (evidence 0.643 accepts a
draft's "0.64"; evidence 45.2 accepts "45%"). A claim with no matching
number anywhere in the evidence is a mismatch -> verdict "fail".

Tool data is the recompute source: every evidence item is a fresh
{tool, args, result} computed from data/ and out/ during this very
investigation, so matching against it IS recomputing from tool data.
"""
from __future__ import annotations

import re

_NUM_RE = re.compile(r"-?\d+(?:\.\d+)?")


def extract_claims(text: str) -> list[str]:
    claims = []
    for m in _NUM_RE.finditer(text):
        if m.start() > 0 and text[m.start() - 1].isalpha():
            continue
        claims.append(m.group(0))
    return claims


def _flatten(obj, acc: list[float]) -> None:
    if isinstance(obj, bool):
        return
    if isinstance(obj, (int, float)):
        acc.append(float(obj))
    elif isinstance(obj, str):
        for s in extract_claims(obj):
            try:
                acc.append(float(s))
            except ValueError:
                continue
    elif isinstance(obj, dict):
        for v in obj.values():
            _flatten(v, acc)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            _flatten(v, acc)


def approved_numbers(evidence: list[dict]) -> list[float]:
    acc: list[float] = []
    for item in evidence:
        if not isinstance(item, dict):
            continue
        _flatten(item.get("args", {}), acc)
        if "result" in item:
            _flatten(item["result"], acc)
    return acc


def _matches(claim: str, approved: list[float]) -> bool:
    try:
        value = float(claim)
    except ValueError:
        return False
    decimals = len(claim.lstrip("-").partition(".")[2])
    tol = 0.5 * (10 ** -decimals) + 1e-9
    return any(abs(v - value) <= tol for v in approved)


def verify(draft: str, evidence: list[dict]) -> dict:
    approved = approved_numbers(evidence)
    claims = list(dict.fromkeys(extract_claims(draft)))
    mismatches = []
    for c in claims:
        if not _matches(c, approved):
            mismatches.append({
                "claim": c,
                "value": float(c),
                "reason": "no matching number in tool evidence",
            })
    return {
        "verdict": "fail" if mismatches else "pass",
        "claims": len(claims),
        "mismatches": mismatches[:20],
    }
