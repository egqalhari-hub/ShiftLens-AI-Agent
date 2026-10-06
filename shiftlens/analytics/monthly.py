"""Monthly analytics -> out/monthly.json (DESIGN.md contract).

Deterministic: reads all data/daily/*.json plus the daily reports' computed
metrics (recomputing from raw data if a report is missing), then emits
team_daily, per-worker aggregates, the 7 correlation pairs with insight
strings, and 3-5 threshold-split findings computed from the data — never
hardcoded.
"""
from __future__ import annotations

import json
import statistics
from pathlib import Path

from shiftlens.agent import metrics as M

_PAIRS = [
    ("fatigue", "units"),
    ("fatigue", "defects"),
    ("noise", "defects"),
    ("temperature", "ppe_ok_pct"),
    ("phone_events", "units"),
    ("mood", "units"),
    ("focus_score", "units"),
]

_TEMPLATES = {
    ("fatigue", "units"): ("Higher fatigue {s} tracks higher output",
                           "Higher fatigue {s} tracks lower output"),
    ("fatigue", "defects"): ("Higher fatigue {s} tracks more defects",
                             "Higher fatigue {s} tracks fewer defects"),
    ("noise", "defects"): ("Louder days {s} track more defects",
                           "Louder days {s} track fewer defects"),
    ("temperature", "ppe_ok_pct"): ("Hotter days {s} track better PPE compliance",
                                    "Hotter days {s} track worse PPE compliance"),
    ("phone_events", "units"): ("More phone events {s} track higher output",
                                "More phone events {s} track lower output"),
    ("mood", "units"): ("Better mood {s} tracks higher output",
                        "Better mood {s} tracks lower output"),
    ("focus_score", "units"): ("Higher focus {s} tracks higher output",
                               "Higher focus {s} tracks lower output"),
}


def _load_daily(data_dir: Path) -> list[dict]:
    files = sorted((data_dir / "daily").glob("*.json"))
    return [json.loads(f.read_text()) for f in files]


def _load_names(data_dir: Path) -> dict[str, str]:
    path = data_dir / "workers.json"
    if not path.exists():
        return {}
    return {w["worker_id"]: w["name"]
            for w in json.loads(path.read_text())["workers"]}


def _load_report_metrics(out_dir: Path) -> dict[tuple[str, str], dict]:
    got: dict[tuple[str, str], dict] = {}
    base = out_dir / "daily_reports"
    if base.is_dir():
        for p in sorted(base.glob("*/*.json")):
            try:
                rep = json.loads(p.read_text())
                got[(rep["date"], rep["worker_id"])] = rep["metrics"]
            except (KeyError, json.JSONDecodeError):
                continue
    return got


def _rows(days: list[dict], report_metrics: dict) -> list[dict]:
    rows = []
    for day in days:
        env = day["environment"]
        for w in day["workers"]:
            m = report_metrics.get((day["date"], w["worker_id"]), {})
            rows.append({
                "date": day["date"],
                "worker_id": w["worker_id"],
                "noise": env["noise_db"],
                "temperature": env["temperature_c"],
                "fatigue": w["survey"]["fatigue"],
                "mood": w["survey"]["mood"],
                "phone_events": w["detections"]["phone_events"],
                "ppe_ok_pct": w["detections"]["ppe_ok_pct"],
                "units": w["output"]["units"],
                "defects": w["output"]["defects"],
                "focus_score": m.get("focus_score", round(M.focus_score(w), 3)),
                "safety_score": m.get("safety_score", round(M.safety_score(w), 3)),
            })
    return rows


def _team_daily(days: list[dict]) -> list[dict]:
    return [{
        "date": day["date"],
        "units": sum(w["output"]["units"] for w in day["workers"]),
        "defects": sum(w["output"]["defects"] for w in day["workers"]),
        "fatigue_avg": round(statistics.fmean(
            w["survey"]["fatigue"] for w in day["workers"]), 2),
        "noise_db": day["environment"]["noise_db"],
    } for day in days]


def _workers_agg(rows: list[dict], names: dict[str, str]) -> list[dict]:
    by: dict[str, list[dict]] = {}
    for r in rows:
        by.setdefault(r["worker_id"], []).append(r)
    out = []
    for wid in sorted(by):
        rs = by[wid]
        units = sum(r["units"] for r in rs)
        defects = sum(r["defects"] for r in rs)
        out.append({
            "worker_id": wid,
            "name": names.get(wid, wid),
            "units_total": units,
            "defect_rate": round(defects / units, 3) if units else 0.0,
            "fatigue_avg": round(statistics.fmean(r["fatigue"] for r in rs), 2),
            "focus_avg": round(statistics.fmean(r["focus_score"] for r in rs), 3),
            "safety_avg": round(statistics.fmean(r["safety_score"] for r in rs), 3),
        })
    return out


def _insight(x: str, y: str, r: float) -> str:
    strength = ("strongly" if abs(r) >= 0.5
                else "moderately" if abs(r) >= 0.25 else "weakly")
    pos, neg = _TEMPLATES[(x, y)]
    return (pos if r >= 0 else neg).format(s=strength)


def _correlations(rows: list[dict]) -> list[dict]:
    out = []
    for x, y in _PAIRS:
        r = M.pearson_r([row[x] for row in rows], [row[y] for row in rows])
        out.append({"x": x, "y": y, "r": round(r, 2), "n": len(rows),
                    "insight": _insight(x, y, r)})
    return out


def _pct(a: float, b: float) -> float:
    return (a - b) / b * 100 if b else 0.0


def _findings(rows: list[dict]) -> list[str]:
    findings = []

    def split(pred, val, min_n=3):
        hi = [val(r) for r in rows if pred(r)]
        lo = [val(r) for r in rows if not pred(r)]
        if len(hi) >= min_n and len(lo) >= min_n:
            return statistics.fmean(hi), statistics.fmean(lo)
        return None

    s = split(lambda r: r["noise"] > 85, lambda r: r["defects"])
    if s:
        a, b = s
        d = _pct(a, b)
        findings.append(
            f"Days above 85 dB average {abs(d):.0f}% "
            f"{'more' if d >= 0 else 'fewer'} defects per worker than quieter "
            f"days ({a:.1f} vs {b:.1f})")

    s = split(lambda r: r["temperature"] > 28, lambda r: r["ppe_ok_pct"])
    if s:
        a, b = s
        d = _pct(a, b)
        findings.append(
            f"Days above 28°C average {abs(d):.0f}% "
            f"{'higher' if d >= 0 else 'lower'} PPE compliance than cooler "
            f"days ({a:.1f}% vs {b:.1f}%)")

    hi = [r["units"] for r in rows if r["fatigue"] >= 4]
    lo = [r["units"] for r in rows if r["fatigue"] <= 2]
    if len(hi) >= 3 and len(lo) >= 3:
        a, b = statistics.fmean(hi), statistics.fmean(lo)
        d = _pct(a, b)
        findings.append(
            f"Worker-days with fatigue ≥ 4 produce {abs(d):.0f}% "
            f"{'more' if d >= 0 else 'fewer'} units than days with fatigue "
            f"≤ 2 ({a:.0f} vs {b:.0f})")

    hi = [r["units"] for r in rows if r["phone_events"] >= 3]
    lo = [r["units"] for r in rows if r["phone_events"] == 0]
    if len(hi) >= 3 and len(lo) >= 3:
        a, b = statistics.fmean(hi), statistics.fmean(lo)
        d = _pct(a, b)
        findings.append(
            f"Worker-days with 3+ phone events produce {abs(d):.0f}% "
            f"{'more' if d >= 0 else 'fewer'} units than zero-event days "
            f"({a:.0f} vs {b:.0f})")

    hi = [r["units"] for r in rows if r["mood"] >= 4]
    lo = [r["units"] for r in rows if r["mood"] <= 2]
    if len(hi) >= 3 and len(lo) >= 3:
        a, b = statistics.fmean(hi), statistics.fmean(lo)
        d = _pct(a, b)
        findings.append(
            f"Worker-days with mood ≥ 4 produce {abs(d):.0f}% "
            f"{'more' if d >= 0 else 'fewer'} units than days with mood "
            f"≤ 2 ({a:.0f} vs {b:.0f})")

    return findings[:5]


def run(data_dir: Path | str, out_dir: Path | str) -> dict:
    data_dir = Path(data_dir)
    out_dir = Path(out_dir)
    days = _load_daily(data_dir)
    if not days:
        raise FileNotFoundError(f"no daily data under {data_dir / 'daily'} — "
                                "run `simulate` first")
    rows = _rows(days, _load_report_metrics(out_dir))
    result = {
        "period": days[0]["date"][:7],
        "team_daily": _team_daily(days),
        "workers": _workers_agg(rows, _load_names(data_dir)),
        "correlations": _correlations(rows),
        "findings": _findings(rows),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "monthly.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return result
