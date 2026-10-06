"""Tool registry (DESIGN.md v2) — the ONLY way agents read data.

`TOOLS = {name: {"fn": callable, "schema": str, "description": str}}`.
Every tool is read-only and returns a JSON-able dict. Tools read from data/
(raw simulated records) and out/ (v1 artifacts: daily reports, monthly.json).

All metric math is reused from shiftlens.agent.metrics and the flag rules
from shiftlens.agent.report — no formula is duplicated here.
"""
from __future__ import annotations

import json
import statistics
from pathlib import Path

import shiftlens
from shiftlens.agent import metrics as M
from shiftlens.agent import report as R

_NOISE_THRESHOLD = 85.0
_TEMP_THRESHOLD = 28.0

_DIRS: dict[str, Path | None] = {"data": None, "out": None}


def configure(data_dir=None, out_dir=None) -> None:
    _DIRS["data"] = Path(data_dir) if data_dir is not None else None
    _DIRS["out"] = Path(out_dir) if out_dir is not None else None


def _data_dir() -> Path:
    return _DIRS["data"] or shiftlens.data_dir()


def _out_dir() -> Path:
    return _DIRS["out"] or shiftlens.out_dir()


def _daily_paths() -> list[Path]:
    return sorted((_data_dir() / "daily").glob("*.json"))


def _load_day(date: str) -> dict:
    path = _data_dir() / "daily" / f"{date}.json"
    if not path.exists():
        raise FileNotFoundError(f"{path} missing — run `simulate` first")
    return json.loads(path.read_text())


def _workers() -> dict[str, dict]:
    path = _data_dir() / "workers.json"
    if not path.exists():
        raise FileNotFoundError(f"{path} missing — run `simulate` first")
    return {w["worker_id"]: w for w in json.loads(path.read_text())["workers"]}


def _means() -> dict[str, dict[str, float]]:
    return R._worker_means(_data_dir())


def _record(day: dict, worker_id: str) -> dict:
    for w in day["workers"]:
        if w["worker_id"] == worker_id:
            return w
    raise ValueError(f"worker {worker_id} not found in {day['date']} data")


def _metrics(wrec: dict) -> dict:
    return {
        "fatigue_index": round(M.fatigue_index(wrec), 3),
        "focus_score": round(M.focus_score(wrec), 3),
        "safety_score": round(M.safety_score(wrec), 3),
        "units": wrec["output"]["units"],
        "defects": wrec["output"]["defects"],
    }


def get_worker_day(date: str, worker_id: str) -> dict:
    day = _load_day(date)
    wrec = _record(day, worker_id)
    worker = _workers().get(worker_id, {})
    means = _means().get(worker_id, {"units": 0.0, "defects": 0.0})
    m = _metrics(wrec)
    return {
        "date": day["date"],
        "worker_id": worker_id,
        "name": worker.get("name", worker_id),
        "role": worker.get("role", "unknown"),
        "shift": worker.get("shift", "day"),
        "environment": day["environment"],
        "sensors": wrec["sensors"],
        "detections": wrec["detections"],
        "survey": wrec["survey"],
        "output": wrec["output"],
        "metrics": m,
        "flags": R._flags(m, means["defects"]),
        "units_avg_30d": round(means["units"], 2),
        "defects_avg_30d": round(means["defects"], 2),
        "days_in_dataset": len(_daily_paths()),
    }


def get_worker_history(worker_id: str, last_n: int = 7,
                       end_date: str | None = None) -> dict:
    paths = _daily_paths()
    if end_date:
        paths = [p for p in paths if p.stem <= end_date]
    try:
        last_n = max(1, int(last_n))
    except (TypeError, ValueError):
        last_n = 7
    paths = paths[-last_n:]
    means = _means().get(worker_id, {"units": 0.0, "defects": 0.0})
    days = []
    for p in paths:
        day = json.loads(p.read_text())
        try:
            wrec = _record(day, worker_id)
        except ValueError:
            continue
        m = _metrics(wrec)
        days.append({"date": day["date"], **m,
                     "flags": R._flags(m, means["defects"])})

    def _avg(key: str):
        vals = [d[key] for d in days]
        return round(statistics.fmean(vals), 3) if vals else None

    return {
        "worker_id": worker_id,
        "window": last_n,
        "end_date": end_date or (paths[-1].stem if paths else None),
        "n_days": len(days),
        "days": days,
        "units_avg": _avg("units"),
        "defects_avg": _avg("defects"),
        "fatigue_avg": _avg("fatigue_index"),
        "focus_avg": _avg("focus_score"),
        "safety_avg": _avg("safety_score"),
    }


_METRIC_KEYS = ("units", "defects", "fatigue_index", "focus_score",
                "safety_score")


def compare_to_team(date: str, metric: str,
                    worker_id: str | None = None) -> dict:
    if metric not in _METRIC_KEYS:
        raise ValueError(
            f"unknown metric {metric!r} — choose from {list(_METRIC_KEYS)}")
    day = _load_day(date)
    per_worker = {w["worker_id"]: _metrics(w)[metric] for w in day["workers"]}
    team_mean = statistics.fmean(per_worker.values())
    out: dict = {
        "date": day["date"],
        "metric": metric,
        "n_workers": len(per_worker),
        "team_mean": round(team_mean, 3),
        "per_worker": {k: (round(v, 3) if isinstance(v, float) else v)
                       for k, v in per_worker.items()},
    }
    if worker_id is not None:
        if worker_id not in per_worker:
            raise ValueError(
                f"worker {worker_id} not found in {day['date']} data")
        wv = per_worker[worker_id]
        delta = wv - team_mean
        pct = (delta / team_mean * 100) if team_mean else 0.0
        out.update({
            "worker_id": worker_id,
            "worker_value": round(wv, 3) if isinstance(wv, float) else wv,
            "delta": round(delta, 3),
            "pct_vs_team": round(pct, 1),
            "pct_abs": round(abs(pct), 1),
            "direction": "above" if pct >= 0 else "below",
        })
    return out


def get_environment_context(date: str) -> dict:
    day = _load_day(date)
    env = day["environment"]
    thresholds = {"noise_db": _NOISE_THRESHOLD, "temperature_c": _TEMP_THRESHOLD}
    crossed = []
    if env["noise_db"] > _NOISE_THRESHOLD:
        crossed.append("noise_db")
    if env["temperature_c"] > _TEMP_THRESHOLD:
        crossed.append("temperature_c")

    loud_def: list[float] = []
    quiet_def: list[float] = []
    hot_ppe: list[float] = []
    cool_ppe: list[float] = []
    n_loud = n_quiet = n_hot = n_cool = 0
    for p in _daily_paths():
        d = json.loads(p.read_text())
        e = d["environment"]
        if e["noise_db"] > _NOISE_THRESHOLD:
            n_loud += 1
            loud_def.extend(w["output"]["defects"] for w in d["workers"])
        else:
            n_quiet += 1
            quiet_def.extend(w["output"]["defects"] for w in d["workers"])
        if e["temperature_c"] > _TEMP_THRESHOLD:
            n_hot += 1
            hot_ppe.extend(w["detections"]["ppe_ok_pct"] for w in d["workers"])
        else:
            n_cool += 1
            cool_ppe.extend(w["detections"]["ppe_ok_pct"] for w in d["workers"])

    def _split(hi, lo, n_hi, n_lo):
        if n_hi < 1 or n_lo < 1:
            return None
        a, b = statistics.fmean(hi), statistics.fmean(lo)
        pct = (a - b) / b * 100 if b else 0.0
        return a, b, pct

    noise = _split(loud_def, quiet_def, n_loud, n_quiet)
    temp = _split(hot_ppe, cool_ppe, n_hot, n_cool)
    defects_total = sum(w["output"]["defects"] for w in day["workers"])
    return {
        "date": day["date"],
        "environment": env,
        "thresholds": thresholds,
        "crossed": crossed,
        "team": {
            "defects_total": defects_total,
            "defects_per_worker": round(defects_total / len(day["workers"]), 2),
            "ppe_avg": round(statistics.fmean(
                w["detections"]["ppe_ok_pct"] for w in day["workers"]), 1),
        },
        "noise_split": ({
            "loud_days": n_loud, "quiet_days": n_quiet,
            "defects_avg_loud": round(noise[0], 2),
            "defects_avg_quiet": round(noise[1], 2),
            "pct_delta": round(noise[2], 1),
            "pct_abs": round(abs(noise[2]), 1),
        } if noise else None),
        "temp_split": ({
            "hot_days": n_hot, "cool_days": n_cool,
            "ppe_avg_hot": round(temp[0], 1),
            "ppe_avg_cool": round(temp[1], 1),
            "pct_delta": round(temp[2], 1),
            "pct_abs": round(abs(temp[2]), 1),
        } if temp else None),
    }


def get_flag_rules() -> dict:
    return {"rules": [
        {"flag": "high_fatigue", "metric": "fatigue_index", "op": ">=",
         "threshold": 0.6,
         "description": "fatigue_index >= 0.6"},
        {"flag": "low_focus", "metric": "focus_score", "op": "<",
         "threshold": 0.7,
         "description": "focus_score < 0.7"},
        {"flag": "safety_risk", "metric": "safety_score", "op": "<",
         "threshold": 0.85,
         "description": "safety_score < 0.85"},
        {"flag": "defect_spike", "metric": "defects", "op": ">=",
         "threshold": 2,
         "description": "defects >= 2x the worker's own mean defects"},
    ]}


def _monthly() -> dict:
    path = _out_dir() / "monthly.json"
    if not path.exists():
        raise FileNotFoundError(f"{path} missing — run `monthly` first")
    return json.loads(path.read_text())


def get_correlations() -> dict:
    m = _monthly()
    return {"period": m.get("period"),
            "correlations": m.get("correlations", [])}


def get_findings() -> dict:
    m = _monthly()
    return {"period": m.get("period"), "findings": m.get("findings", [])}


def get_flagged_days(worker_id: str) -> dict:
    paths = _daily_paths()
    means = _means()
    if worker_id not in means:
        raise ValueError(f"worker {worker_id} not found in any daily data")
    dates = []
    for p in paths:
        day = json.loads(p.read_text())
        try:
            wrec = _record(day, worker_id)
        except ValueError:
            continue
        if R._flags(_metrics(wrec), means[worker_id]["defects"]):
            dates.append(day["date"])
    return {"worker_id": worker_id, "n_days": len(paths),
            "n_flagged": len(dates), "dates": dates}


TOOLS = {
    "get_worker_day": {
        "fn": get_worker_day,
        "schema": "get_worker_day(date: str, worker_id: str) -> dict",
        "description": "Raw record + computed metrics + flags for one "
                       "worker-day.",
    },
    "get_worker_history": {
        "fn": get_worker_history,
        "schema": "get_worker_history(worker_id: str, last_n: int = 7, "
                  "end_date: str | None = None) -> dict",
        "description": "Per-day metrics/units/defects over the worker's "
                       "recent days.",
    },
    "compare_to_team": {
        "fn": compare_to_team,
        "schema": "compare_to_team(date: str, metric: str, "
                  "worker_id: str | None = None) -> dict",
        "description": "Worker value vs team mean for one metric on one day "
                       "(metric in units/defects/fatigue_index/focus_score/"
                       "safety_score).",
    },
    "get_environment_context": {
        "fn": get_environment_context,
        "schema": "get_environment_context(date: str) -> dict",
        "description": "Env readings, crossed thresholds (noise>85 dB, "
                       "temp>28 C) and team defect/PPE deltas on those days.",
    },
    "get_flag_rules": {
        "fn": get_flag_rules,
        "schema": "get_flag_rules() -> dict",
        "description": "The deterministic flag rules (planner context).",
    },
    "get_correlations": {
        "fn": get_correlations,
        "schema": "get_correlations() -> dict",
        "description": "Monthly analytics correlations (requires "
                       "out/monthly.json).",
    },
    "get_findings": {
        "fn": get_findings,
        "schema": "get_findings() -> dict",
        "description": "Monthly analytics findings (requires "
                       "out/monthly.json).",
    },
    "get_flagged_days": {
        "fn": get_flagged_days,
        "schema": "get_flagged_days(worker_id: str) -> dict",
        "description": "All dates where the worker had any flag.",
    },
}
