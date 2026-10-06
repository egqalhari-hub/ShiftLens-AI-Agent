"""ShiftLens — agentic workforce analytics (see DESIGN.md, the binding contract).

The LLM phrases; it never decides. All metrics, flags, correlations and
findings are deterministic Python (stdlib only).
"""
from __future__ import annotations

import os
from pathlib import Path

__version__ = "0.1.0"

REPO_ROOT = Path(__file__).resolve().parent.parent


def data_dir() -> Path:
    """Dataset directory (workers.json + daily/). Overridable for tests."""
    return Path(os.environ.get("SHIFTLENS_DATA_DIR", str(REPO_ROOT / "data")))


def out_dir() -> Path:
    """Output directory (daily_reports/ + monthly.json). Overridable for tests."""
    return Path(os.environ.get("SHIFTLENS_OUT_DIR", str(REPO_ROOT / "out")))
