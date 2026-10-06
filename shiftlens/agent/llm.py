"""Optional LM Studio phrasing layer (OpenAI-compatible chat API).

The LLM only rewrites the daily report's prose summary — it never decides
anything. Every failure mode (server down, timeout, bad payload) returns
None so callers fall back to the deterministic template. This module must
never crash the pipeline.

Config: SHIFTLENS_LMS_HOST (default http://localhost:1234),
        SHIFTLENS_LMS_MODEL (default bonsai-27b),
        SHIFTLENS_LMS_TIMEOUT (seconds, default 30).
"""
from __future__ import annotations

import json
import os
import urllib.request

HOST = os.environ.get("SHIFTLENS_LMS_HOST", "http://localhost:1234")
MODEL = os.environ.get("SHIFTLENS_LMS_MODEL", "bonsai-27b")
TIMEOUT = float(os.environ.get("SHIFTLENS_LMS_TIMEOUT", "30"))

_SYSTEM = (
    "You rewrite one worker's deterministic ShiftLens daily-report summary. "
    "Hard rules: NEVER invent, add, drop, or alter any number — every number "
    "in your output must appear verbatim in the JSON facts provided. Use only "
    "those facts. 2-4 sentences, neutral professional tone, no advice beyond "
    "the flags listed."
)


def summarize(context: dict, timeout: float | None = None) -> str | None:
    """Return an LLM-rephrased summary, or None if LM Studio is unreachable."""
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": _SYSTEM},
            {"role": "user", "content":
                "Rewrite these daily-report facts as prose:\n"
                + json.dumps(context, indent=2)},
        ],
        "temperature": 0.2,
        "max_tokens": 220,
    }
    try:
        req = urllib.request.Request(
            HOST.rstrip("/") + "/v1/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(
                req, timeout=TIMEOUT if timeout is None else timeout) as resp:
            body = json.loads(resp.read().decode("utf-8"))
        text = body["choices"][0]["message"]["content"].strip()
        return text or None
    except Exception:
        return None
