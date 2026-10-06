"""Deterministic sensor/survey/output simulator (DESIGN.md contract).

Writes data/workers.json and data/daily/YYYY-MM-DD.json for 8 workers over
30 days starting 2026-09-01, using a single random.Random(42) stream so the
whole dataset is byte-for-byte reproducible.

Hidden causal structure (analytics must rediscover these, never hardcoded):
  * noise_db > 85 days          -> defect rate rises
  * survey fatigue up           -> units down, defects up
  * temperature_c > 28          -> ppe_ok_pct drops
  * phone_events up (more likely on low-mood days) -> units down
  * per-worker base_skill scales units; small daily noise everywhere
"""
from __future__ import annotations

import json
import random
from datetime import date, timedelta
from pathlib import Path

SEED = 42
N_WORKERS = 8
N_DAYS = 30
START = date(2026, 9, 1)

_NAMES = [
    "Aria Kim", "Mateo Ruiz", "Priya Nair", "Jonas Weber",
    "Sofia Rossi", "Daniel Okafor", "Mei Tanaka", "Lucas Silva",
]
_ROLES = ["assembler", "packer", "inspector", "forklift-operator"]
_BLOCKERS = [
    "machine downtime", "missing parts", "waiting on forklift",
    "unclear instructions", "supplier delay",
]


def _build_workers(rng: random.Random) -> list[dict]:
    workers = []
    for i in range(N_WORKERS):
        workers.append({
            "worker_id": f"W{i + 1:02d}",
            "name": _NAMES[i],
            "role": _ROLES[i % len(_ROLES)],
            "shift": "night" if i in (3, 7) else "day",
            "base_skill": round(rng.uniform(0.85, 1.20), 2),
        })
    return workers


def _environment(rng: random.Random) -> dict:
    if rng.random() < 0.25:
        noise_db = rng.uniform(86.0, 93.0)
    else:
        noise_db = rng.uniform(72.0, 84.5)
    if rng.random() < 0.20:
        temperature_c = rng.uniform(28.3, 31.5)
    else:
        temperature_c = rng.uniform(22.5, 27.5)
    return {
        "noise_db": round(noise_db, 1),
        "temperature_c": round(temperature_c, 1),
        "air_quality_pm25": int(round(rng.uniform(5, 28))),
    }


def _worker_day(rng: random.Random, worker: dict, env: dict) -> dict:
    fatigue = int(round(rng.gauss(2.6, 1.0)))
    if env["noise_db"] > 85 and rng.random() < 0.5:
        fatigue += 1
    fatigue = max(1, min(5, fatigue))
    mood = max(1, min(5, int(round(5.4 - 0.75 * fatigue + rng.gauss(0, 0.6)))))
    phone_events = max(0, int(round(rng.gauss(0.4 + 0.9 * (5 - mood), 0.9))))
    idle_events = max(0, int(round(rng.gauss(1.2 + 0.55 * fatigue, 1.0))))
    r = rng.random()
    violations = 2 if r < 0.02 else (1 if r < 0.10 else 0)
    ppe_ok_pct = rng.uniform(94.0, 99.5)
    if env["temperature_c"] > 28:
        ppe_ok_pct -= rng.uniform(8.0, 18.0)
    ppe_ok_pct = round(max(55.0, min(100.0, ppe_ok_pct)), 1)
    hr_avg = int(round(64 + 4.5 * fatigue + rng.gauss(0, 4)))
    hr_max = hr_avg + int(round(rng.uniform(24, 46)))
    steps = int(round(5600 + 900 * worker["base_skill"] + rng.gauss(0, 700)))
    station_time_min = int(round(430 - 14 * fatigue + rng.gauss(0, 18)))
    units = int(round(
        worker["base_skill"] * (165 - 9 * fatigue - 5 * phone_events)
        + rng.gauss(0, 6)
    ))
    units = max(10, units)
    defect_lam = 1.0 + 0.55 * fatigue + (3.0 if env["noise_db"] > 85 else 0.0)
    defects = max(0, int(round(rng.gauss(defect_lam, 1.0))))
    tasks_completed = max(1, int(round(units / 16 + rng.gauss(0, 1))))
    perceived = max(1, min(5, int(round(units / 35))))
    blockers = ""
    if fatigue >= 4 and rng.random() < 0.4:
        blockers = rng.choice(_BLOCKERS)
    return {
        "worker_id": worker["worker_id"],
        "sensors": {"hr_avg": hr_avg, "hr_max": hr_max, "steps": steps,
                    "station_time_min": station_time_min},
        "detections": {"ppe_ok_pct": ppe_ok_pct, "phone_events": phone_events,
                       "idle_events": idle_events,
                       "safety_zone_violations": violations},
        "survey": {"fatigue": fatigue, "mood": mood,
                   "perceived_productivity": perceived, "blockers": blockers},
        "output": {"units": units, "defects": defects,
                   "tasks_completed": tasks_completed},
    }


def run(data_dir: Path | str) -> dict:
    """Generate the full deterministic dataset. Returns a small summary."""
    rng = random.Random(SEED)
    data_dir = Path(data_dir)
    daily_dir = data_dir / "daily"
    daily_dir.mkdir(parents=True, exist_ok=True)
    workers = _build_workers(rng)
    (data_dir / "workers.json").write_text(
        json.dumps({"workers": workers}, indent=2) + "\n", encoding="utf-8")
    for i in range(N_DAYS):
        day = START + timedelta(days=i)
        env = _environment(rng)
        records = [_worker_day(rng, w, env) for w in workers]
        payload = {"date": day.isoformat(), "environment": env, "workers": records}
        (daily_dir / f"{day.isoformat()}.json").write_text(
            json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return {"workers": len(workers), "days": N_DAYS, "start": START.isoformat()}
