"""Deterministic per-worker daily reports -> out/daily_reports/DATE/WID.json + .md

Metrics and flags are pure Python (agent/metrics.py). The optional LLM only
rephrases the prose summary; if it is unreachable or drops the numbers, the
deterministic template is used instead. Summaries ALWAYS list the numbers.
"""
from __future__ import annotations

import json
from pathlib import Path

from shiftlens.agent import llm, metrics as M


def _daily_files(data_dir: Path) -> list[Path]:
    return sorted((data_dir / "daily").glob("*.json"))


def _load_workers(data_dir: Path) -> dict[str, dict]:
    path = data_dir / "workers.json"
    if not path.exists():
        raise FileNotFoundError(f"{path} missing — run `simulate` first")
    return {w["worker_id"]: w for w in json.loads(path.read_text())["workers"]}


def _worker_means(data_dir: Path) -> dict[str, dict[str, float]]:
    """Per-worker mean units/defects across all available daily files."""
    totals: dict[str, list[float]] = {}
    for f in _daily_files(data_dir):
        day = json.loads(f.read_text())
        for w in day["workers"]:
            t = totals.setdefault(w["worker_id"], [0.0, 0.0, 0.0])
            t[0] += w["output"]["units"]
            t[1] += w["output"]["defects"]
            t[2] += 1
    return {wid: {"units": u / n, "defects": d / n}
            for wid, (u, d, n) in totals.items() if n}


def _flags(m: dict, worker_mean_defects: float) -> list[str]:
    """DESIGN.md flag rules, in fixed order."""
    flags = []
    if m["fatigue_index"] >= 0.6:
        flags.append("high_fatigue")
    if m["focus_score"] < 0.7:
        flags.append("low_focus")
    if m["safety_score"] < 0.85:
        flags.append("safety_risk")
    if worker_mean_defects > 0 and m["defects"] >= 2 * worker_mean_defects:
        flags.append("defect_spike")
    return flags


def _template_summary(worker: dict, date: str, m: dict, flags: list[str],
                      mean_units: float) -> str:
    pct = ((m["units"] - mean_units) / mean_units * 100) if mean_units else 0.0
    trend = "above" if pct >= 0 else "below"
    flag_txt = ", ".join(flags) if flags else "none"
    return (
        f"{worker['name']} ({worker['worker_id']}, {worker['role']}, "
        f"{worker['shift']} shift) produced {m['units']} units with "
        f"{m['defects']} defects on {date} ({abs(pct):.0f}% {trend} their "
        f"30-day average of {mean_units:.0f} units). Fatigue index "
        f"{m['fatigue_index']:.2f}, focus score {m['focus_score']:.2f}, "
        f"safety score {m['safety_score']:.2f}. Flags: {flag_txt}."
    )


def build_report(worker: dict, date: str, wrec: dict,
                 means: dict[str, dict[str, float]]) -> dict:
    """One worker-day report dict, per the DESIGN.md schema."""
    wid = wrec["worker_id"]
    wm = means.get(wid, {"units": 0.0, "defects": 0.0})
    m = {
        "fatigue_index": round(M.fatigue_index(wrec), 3),
        "focus_score": round(M.focus_score(wrec), 3),
        "safety_score": round(M.safety_score(wrec), 3),
        "units": wrec["output"]["units"],
        "defects": wrec["output"]["defects"],
    }
    flags = _flags(m, wm["defects"])

    context = {
        "worker": worker["name"], "worker_id": wid, "role": worker["role"],
        "shift": worker["shift"], "date": date, "metrics": m, "flags": flags,
        "units_avg_30d": round(wm["units"], 1),
    }
    summary = llm.summarize(context)
    if not summary or str(m["units"]) not in summary:
        summary = _template_summary(worker, date, m, flags, wm["units"])

    return {"worker_id": wid, "date": date, "metrics": m,
            "flags": flags, "summary": summary}


def _markdown(rep: dict, worker: dict) -> str:
    m = rep["metrics"]
    flags = "\n".join(f"- {f}" for f in rep["flags"]) if rep["flags"] else "None"
    return (
        f"# Daily report — {worker['name']} ({rep['worker_id']}) — {rep['date']}\n\n"
        f"Role: {worker['role']} · Shift: {worker['shift']}\n\n"
        "## Metrics\n\n"
        "| metric | value |\n"
        "| --- | --- |\n"
        f"| units | {m['units']} |\n"
        f"| defects | {m['defects']} |\n"
        f"| fatigue_index | {m['fatigue_index']:.2f} |\n"
        f"| focus_score | {m['focus_score']:.2f} |\n"
        f"| safety_score | {m['safety_score']:.2f} |\n\n"
        f"## Flags\n\n{flags}\n\n"
        f"## Summary\n\n{rep['summary']}\n"
    )


def run(data_dir: Path | str, out_dir: Path | str, date: str | None = None) -> int:
    """Generate reports for one date (or all dates). Returns report count."""
    data_dir = Path(data_dir)
    out_dir = Path(out_dir)
    workers = _load_workers(data_dir)
    means = _worker_means(data_dir)
    if not means:
        raise FileNotFoundError(f"no daily data under {data_dir / 'daily'} — "
                                "run `simulate` first")

    files = _daily_files(data_dir)
    if date is not None:
        files = [f for f in files if f.stem == date]
        if not files:
            raise FileNotFoundError(f"no daily data for date {date}")

    count = 0
    for f in files:
        day = json.loads(f.read_text())
        day_dir = out_dir / "daily_reports" / day["date"]
        day_dir.mkdir(parents=True, exist_ok=True)
        for wrec in day["workers"]:
            worker = workers.get(wrec["worker_id"], {
                "worker_id": wrec["worker_id"], "name": wrec["worker_id"],
                "role": "unknown", "shift": "day"})
            rep = build_report(worker, day["date"], wrec, means)
            wid = rep["worker_id"]
            (day_dir / f"{wid}.json").write_text(
                json.dumps(rep, indent=2) + "\n", encoding="utf-8")
            (day_dir / f"{wid}.md").write_text(_markdown(rep, worker),
                                               encoding="utf-8")
            count += 1
    return count
