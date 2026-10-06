"""Role agents (DESIGN.md v2 roster).

- orchestrator — lives in agents/orchestrator.py (runs the loop, writes traces)
- planner      — plan(task): ordered tool calls; LLM if reachable, else the
                 deterministic flag-driven heuristic plan
- investigator — investigate(plan): pure executor, collects evidence items
                 {tool, args, result}; no prose
- drafter      — draft()/revise(): prose from evidence only; LLM may phrase,
                 the deterministic template is always the fallback
- verifier     — lives in agents/verifier.py (the critic)

The LLM only ever phrases/plans; it never decides. LLM calls reuse
shiftlens.agent.llm's host/env config with short timeouts, and any failure
disables the LLM for the rest of the process (offline-first: the full loop
runs on heuristic plans + template drafts with LM Studio down).
"""
from __future__ import annotations

import inspect
import json
import urllib.request

from shiftlens.agent import llm as _lms
from shiftlens.agents import tools as T

ROLES = {
    "orchestrator": "runs plan -> investigate -> draft -> verify -> "
                    "(revise once) -> publish; writes the trace",
    "planner": "emits an investigation plan: an ordered list of tool calls",
    "investigator": "executes tool calls, collects evidence items "
                    "{tool, args, result}",
    "drafter": "writes prose from evidence only; every number must come "
               "from evidence",
    "verifier": "re-checks every numeric claim in the draft against tool data",
}

_FLAG_METRIC = {
    "high_fatigue": "fatigue_index",
    "low_focus": "focus_score",
    "safety_risk": "safety_score",
    "defect_spike": "defects",
}

_MAX_PLAN_CALLS = 10
_LLM_TIMEOUT = 15.0

_LLM = {"available": None, "used": 0}


def llm_used_count() -> int:
    """Successful LLM calls so far (orchestrator derives per-trace llm_used)."""
    return _LLM["used"]


def _chat(system: str, user: str, max_tokens: int = 500) -> str | None:
    """One OpenAI-compatible chat call against LM Studio; None on any
    failure (server down, timeout, bad payload). Never raises."""
    if _LLM["available"] is False:
        return None
    payload = {
        "model": _lms.MODEL,
        "messages": [{"role": "system", "content": system},
                     {"role": "user", "content": user}],
        "temperature": 0.2,
        "max_tokens": max_tokens,
    }
    try:
        req = urllib.request.Request(
            _lms.HOST.rstrip("/") + "/v1/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(
                req, timeout=min(_lms.TIMEOUT, _LLM_TIMEOUT)) as resp:
            body = json.loads(resp.read().decode("utf-8"))
        text = body["choices"][0]["message"]["content"].strip()
        if not text:
            return None
        _LLM["available"] = True
        _LLM["used"] += 1
        return text
    except Exception:
        _LLM["available"] = False
        return None


_PLAN_SYSTEM = (
    "You are the ShiftLens planner agent. Given one task, output ONLY a JSON "
    "array of tool calls — no prose, no code fences — e.g.\n"
    '[{"tool": "get_worker_day", "args": {"date": "2026-09-01", '
    '"worker_id": "W01"}}]\n'
    "Available tools:\n" + "\n".join(
        f"- {name}: {spec['schema']} — {spec['description']}"
        for name, spec in T.TOOLS.items()) + "\n"
    "Rules: a worker-day investigation starts with get_worker_day and ends "
    "with get_flagged_days; add get_worker_history(worker_id, 7, end_date); "
    "add compare_to_team for the metric behind each flag (high_fatigue -> "
    "fatigue_index, low_focus -> focus_score, safety_risk -> safety_score, "
    "defect_spike -> defects); add get_environment_context if an environment "
    "threshold was crossed. A monthly_review task uses get_correlations and "
    "get_findings."
)


def heuristic_plan(task: dict) -> list[dict]:
    if task.get("type") == "monthly_review":
        return [{"tool": "get_correlations", "args": {}},
                {"tool": "get_findings", "args": {}}]
    date, wid = task["date"], task["worker_id"]
    calls = [{"tool": "get_worker_day",
              "args": {"date": date, "worker_id": wid}}]
    seen_metrics = set()
    for flag in task.get("flags", []):
        metric = _FLAG_METRIC.get(flag)
        calls.append({"tool": "get_worker_history",
                      "args": {"worker_id": wid, "last_n": 7,
                               "end_date": date}})
        if metric and metric not in seen_metrics:
            seen_metrics.add(metric)
            calls.append({"tool": "compare_to_team",
                          "args": {"date": date, "metric": metric,
                                   "worker_id": wid}})
    try:
        env = T.TOOLS["get_environment_context"]["fn"](date)
        crossed = env.get("crossed", [])
    except Exception:
        crossed = []
    if crossed:
        calls.append({"tool": "get_environment_context",
                      "args": {"date": date}})
    calls.append({"tool": "get_flagged_days", "args": {"worker_id": wid}})
    return calls


def _parse_llm_plan(text: str) -> list[dict] | None:
    i, j = text.find("["), text.rfind("]")
    if i < 0 or j <= i:
        return None
    try:
        raw = json.loads(text[i:j + 1])
    except (json.JSONDecodeError, ValueError):
        return None
    if not isinstance(raw, list):
        return None
    calls = []
    for item in raw:
        if (isinstance(item, dict) and item.get("tool") in T.TOOLS
                and isinstance(item.get("args", {}), dict)):
            calls.append({"tool": item["tool"], "args": item.get("args", {})})
    return calls or None


def _normalize_plan(task: dict, calls: list[dict]) -> list[dict]:
    seen = set()
    out = []
    for c in calls:
        key = (c["tool"], json.dumps(c.get("args", {}), sort_keys=True,
                                     default=str))
        if key in seen:
            continue
        seen.add(key)
        out.append({"tool": c["tool"], "args": c.get("args", {})})
        if len(out) >= _MAX_PLAN_CALLS:
            break
    if task.get("type") == "investigation":
        if not any(c["tool"] == "get_worker_day" for c in out):
            out.insert(0, {"tool": "get_worker_day",
                           "args": {"date": task["date"],
                                    "worker_id": task["worker_id"]}})
        if not any(c["tool"] == "get_flagged_days" for c in out):
            out.append({"tool": "get_flagged_days",
                        "args": {"worker_id": task["worker_id"]}})
    elif task.get("type") == "monthly_review":
        for need in ("get_correlations", "get_findings"):
            if not any(c["tool"] == need for c in out):
                out.append({"tool": need, "args": {}})
    return out


def plan(task: dict) -> list[dict]:
    calls = None
    text = _chat(_PLAN_SYSTEM, json.dumps(task), max_tokens=600)
    if text:
        calls = _parse_llm_plan(text)
    if not calls:
        calls = heuristic_plan(task)
    return _normalize_plan(task, calls)


def _call_tool(name: str, args: dict):
    fn = T.TOOLS[name]["fn"]
    params = inspect.signature(fn).parameters
    clean = {k: v for k, v in args.items() if k in params}
    return fn(**clean)


def investigate(plan: list[dict]) -> list[dict]:
    evidence = []
    for call in plan:
        name, args = call["tool"], call.get("args", {})
        try:
            evidence.append({"tool": name, "args": args,
                             "result": _call_tool(name, args)})
        except Exception as exc:
            evidence.append({"tool": name, "args": args,
                             "error": f"{type(exc).__name__}: {exc}"})
    return evidence


_DRAFT_SYSTEM = (
    "You are the ShiftLens drafter agent. Write a concise investigation note "
    "(3-6 sentences) for a shift supervisor using ONLY the supplied tool "
    "evidence. HARD RULE: every number in your note must appear in the "
    "evidence JSON (rounding is allowed); never invent, drop the sign of, "
    "or extrapolate numbers. Neutral, factual tone."
)

_NARRATIVE_SYSTEM = (
    "You are the ShiftLens drafter agent writing the editor's monthly "
    "review (4-8 sentences) from tool evidence only. HARD RULE: every "
    "number in your narrative must appear in the evidence JSON (rounding "
    "is allowed); never invent numbers. Neutral, factual tone."
)

_REVISE_SYSTEM = (
    "You are the ShiftLens drafter agent. The verifier rejected your draft "
    "because these numeric claims are NOT supported by the tool evidence: "
    "{mismatches}. Rewrite the draft so every number appears in the "
    "evidence JSON. Output only the revised prose."
)


def _find(evidence: list[dict], tool: str) -> dict | None:
    for item in evidence:
        if item.get("tool") == tool and "result" in item:
            return item["result"]
    return None


def _find_all(evidence: list[dict], tool: str) -> list[dict]:
    return [i["result"] for i in evidence
            if i.get("tool") == tool and "result" in i]


def likely_drivers(task: dict, evidence: list[dict]) -> list[str]:
    wd = _find(evidence, "get_worker_day")
    if not wd or "metrics" not in wd:
        return ["tool evidence unavailable"]
    m, det = wd["metrics"], wd["detections"]
    by_metric = {c.get("metric"): c
                 for c in _find_all(evidence, "compare_to_team")}
    drivers = []
    for flag in task.get("flags", []):
        if flag == "high_fatigue":
            c = by_metric.get("fatigue_index")
            if c:
                drivers.append(
                    f"elevated fatigue (index {c['worker_value']:.2f} vs "
                    f"team mean {c['team_mean']:.2f})")
            else:
                drivers.append(
                    f"elevated fatigue (index {m['fatigue_index']:.2f})")
        elif flag == "low_focus":
            drivers.append(
                f"low focus (score {m['focus_score']:.2f}; "
                f"{det['phone_events']} phone events, "
                f"{det['idle_events']} idle events)")
        elif flag == "safety_risk":
            drivers.append(
                f"safety risk (score {m['safety_score']:.2f}; PPE ok "
                f"{det['ppe_ok_pct']}%, "
                f"{det['safety_zone_violations']} zone violations)")
        elif flag == "defect_spike":
            drivers.append(
                f"defect spike ({m['defects']} defects vs 30-day average "
                f"{wd['defects_avg_30d']:.1f})")
    env = _find(evidence, "get_environment_context")
    if env and env.get("crossed"):
        e, thr = env["environment"], env["thresholds"]
        if "noise_db" in env["crossed"]:
            drivers.append(
                f"loud environment ({e['noise_db']} dB > "
                f"{thr['noise_db']:g} dB threshold)")
        if "temperature_c" in env["crossed"]:
            drivers.append(
                f"hot environment ({e['temperature_c']}°C > "
                f"{thr['temperature_c']:g}°C threshold)")
    fl = _find(evidence, "get_flagged_days")
    if fl and fl.get("n_flagged", 0) > 1:
        drivers.append(
            f"recurring issue ({fl['n_flagged']} flagged days this month)")
    return drivers or ["no dominant driver identified in tool evidence"]


def template_note(task: dict, evidence: list[dict]) -> str:
    wd = _find(evidence, "get_worker_day")
    if not wd or "metrics" not in wd:
        return (f"Investigation of {task.get('worker_id')} on "
                f"{task.get('date')}: tool evidence unavailable; "
                "no numeric claims made.")
    m = wd["metrics"]
    flags = ", ".join(task.get("flags", [])) or "none"
    parts = [
        f"Investigation of {task['worker_id']} on {task['date']} "
        f"(flags: {flags}). Output was {m['units']} units with "
        f"{m['defects']} defects; fatigue index {m['fatigue_index']:.2f}, "
        f"focus score {m['focus_score']:.2f}, safety score "
        f"{m['safety_score']:.2f}."
    ]
    hist = _find(evidence, "get_worker_history")
    if hist and hist.get("n_days"):
        parts.append(
            f"Over the worker's last {hist['n_days']} day(s) up to "
            f"{hist['end_date']}: average units {hist['units_avg']:.1f}, "
            f"defects {hist['defects_avg']:.1f}, fatigue index "
            f"{hist['fatigue_avg']:.2f}.")
    for c in _find_all(evidence, "compare_to_team"):
        parts.append(
            f"On {c['date']}, {c['metric']} was {c['worker_value']} vs "
            f"team mean {c['team_mean']:.2f} — {c['pct_abs']:.0f}% "
            f"{c['direction']}.")
    env = _find(evidence, "get_environment_context")
    if env and env.get("crossed"):
        e, thr = env["environment"], env["thresholds"]
        bits = []
        if "noise_db" in env["crossed"]:
            bits.append(f"noise {e['noise_db']} dB crossed the "
                        f"{thr['noise_db']:g} dB threshold")
            ns = env.get("noise_split")
            if ns:
                bits.append(
                    f"loud days this month average {ns['pct_abs']:.0f}% "
                    f"{'more' if ns['pct_delta'] >= 0 else 'fewer'} defects "
                    f"per worker ({ns['defects_avg_loud']:.1f} vs "
                    f"{ns['defects_avg_quiet']:.1f})")
        if "temperature_c" in env["crossed"]:
            bits.append(f"temperature {e['temperature_c']}°C crossed the "
                        f"{thr['temperature_c']:g}°C threshold")
            ts = env.get("temp_split")
            if ts:
                bits.append(
                    f"hot days average {ts['pct_abs']:.0f}% "
                    f"{'lower' if ts['pct_delta'] <= 0 else 'higher'} PPE "
                    f"compliance ({ts['ppe_avg_hot']:.1f}% vs "
                    f"{ts['ppe_avg_cool']:.1f}%)")
        parts.append("Environment: " + "; ".join(bits) + ".")
    fl = _find(evidence, "get_flagged_days")
    if fl:
        parts.append(
            f"Recurrence: {fl['n_flagged']} of {fl['n_days']} days flagged "
            f"for this worker this month.")
    parts.append("Likely drivers: "
                 + "; ".join(likely_drivers(task, evidence)) + ".")
    return " ".join(parts)


def template_narrative(task: dict, evidence: list[dict]) -> str:
    corr = _find(evidence, "get_correlations")
    if not corr or not corr.get("correlations"):
        return ("Monthly review: analytics evidence unavailable; "
                "no numeric claims made.")
    period = corr.get("period") or task.get("period")
    cl = corr["correlations"]
    n = cl[0].get("n", 0)
    parts = [f"Month review — {period}: {n} worker-days analysed."]
    top = sorted(cl, key=lambda c: -abs(c.get("r", 0)))[:3]
    for c in top:
        parts.append(
            f"{c['insight']} ({c['x']} vs {c['y']}, r={c['r']:+.2f}).")
    findings = (_find(evidence, "get_findings") or {}).get("findings", [])
    if findings:
        parts.append("Key findings: " + " ".join(
            f if f.rstrip().endswith(".") else f + "." for f in findings))
    parts.append("Every figure above comes from the deterministic analytics "
                 "and was re-checked against tool data by the verifier "
                 "agent.")
    return " ".join(parts)


def draft(task: dict, evidence: list[dict]) -> dict:
    is_monthly = task.get("type") == "monthly_review"
    drivers = [] if is_monthly else likely_drivers(task, evidence)
    system = _NARRATIVE_SYSTEM if is_monthly else _DRAFT_SYSTEM
    payload = {"task": task, "likely_drivers": drivers, "evidence": evidence}
    text = _chat(system, json.dumps(payload, default=str), max_tokens=600)
    if text and text.strip():
        return {"note": text.strip(), "likely_drivers": drivers,
                "source": "llm"}
    fallback = (template_narrative(task, evidence) if is_monthly
                else template_note(task, evidence))
    return {"note": fallback, "likely_drivers": drivers,
            "source": "template"}


def revise(task: dict, evidence: list[dict], draft_text: str,
           mismatches: list[dict]) -> dict:
    is_monthly = task.get("type") == "monthly_review"
    drivers = [] if is_monthly else likely_drivers(task, evidence)
    bad = ", ".join(m["claim"] for m in mismatches) or "unsupported numbers"
    payload = {"task": task, "draft": draft_text,
               "mismatches": mismatches, "evidence": evidence}
    text = _chat(_REVISE_SYSTEM.format(mismatches=bad),
                 json.dumps(payload, default=str), max_tokens=600)
    if text and text.strip():
        return {"note": text.strip(), "likely_drivers": drivers,
                "source": "llm_revision"}
    fallback = (template_narrative(task, evidence) if is_monthly
                else template_note(task, evidence))
    return {"note": fallback, "likely_drivers": drivers,
            "source": "template_revision"}
