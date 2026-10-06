"""Pure metric functions — formulas fixed by DESIGN.md. Stdlib only."""
from __future__ import annotations

import math
import statistics


def clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


def fatigue_index(rec: dict) -> float:
    """0.6*(survey.fatigue/5) + 0.4*min(hr_avg/100, 1.2)/1.2"""
    fatigue = rec["survey"]["fatigue"]
    hr_avg = rec["sensors"]["hr_avg"]
    return 0.6 * (fatigue / 5) + 0.4 * min(hr_avg / 100, 1.2) / 1.2


def focus_score(rec: dict) -> float:
    """clamp(1 - 0.08*phone_events - 0.03*idle_events, 0, 1)"""
    det = rec["detections"]
    return clamp(1 - 0.08 * det["phone_events"] - 0.03 * det["idle_events"])


def safety_score(rec: dict) -> float:
    """clamp(ppe_ok_pct/100 - 0.25*safety_zone_violations, 0, 1)"""
    det = rec["detections"]
    return clamp(det["ppe_ok_pct"] / 100 - 0.25 * det["safety_zone_violations"])


def productivity_normalized(units: float, worker_mean_units: float) -> float:
    """Units normalized per worker vs their own 30-day mean."""
    if not worker_mean_units:
        return 0.0
    return units / worker_mean_units


def pearson_r(xs, ys) -> float:
    """Pearson correlation coefficient via stdlib statistics. 0.0 if degenerate."""
    xs = list(xs)
    ys = list(ys)
    n = len(xs)
    if n < 2 or n != len(ys):
        return 0.0
    mx = statistics.fmean(xs)
    my = statistics.fmean(ys)
    sxx = syy = sxy = 0.0
    for x, y in zip(xs, ys):
        dx = x - mx
        dy = y - my
        sxx += dx * dx
        syy += dy * dy
        sxy += dx * dy
    if sxx <= 0 or syy <= 0:
        return 0.0
    return sxy / math.sqrt(sxx * syy)
