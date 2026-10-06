"""Orchestrator agent (DESIGN.md v2): runs the full loop per task and writes
the judge-facing trace.

    plan -> investigate -> draft -> verify -> (revise once) -> publish

- run_investigation(date, worker_id): investigates one flagged worker-day;
  the daily report gains an "investigation" block (JSON + markdown section,
  updated in place) and a trace lands in out/traces/<date>_<WID>.json.
- run_monthly_review(): reads out/monthly.json (written by monthly.run()),
  drafts + verifies the "narrative", adds the "agentic" summary and writes
  monthly.json back in place; trace lands in out/traces/monthly.json.

On a double verifier fail the deterministic template text is published and
the trace records it (published_source = "template_fallback"). v1 fields are
never modified.
"""
from __future__ import annotations

import json
from pathlib import Path

import shiftlens
from shiftlens.agents import roles, tools, verifier


def _dirs(data_dir, out_dir) -> tuple[Path, Path]:
    tools.configure(data_dir, out_dir)
    dd = Path(data_dir) if data_dir is not None else shiftlens.data_dir()
    od = Path(out_dir) if out_dir is not None else shiftlens.out_dir()
    return dd, od


def find_flagged_worker_days(out_dir=None,
                             date: str | None = None) -> list[tuple[str, str]]:
    od = Path(out_dir) if out_dir is not None else shiftlens.out_dir()
    base = od / "daily_reports"
    pairs = set()
    for p in sorted(base.glob("*/*.json")):
        if date is not None and p.parent.name != date:
            continue
        try:
            rep = json.loads(p.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        if rep.get("flags"):
            pairs.add((rep["date"], rep["worker_id"]))
    return sorted(pairs)


def _write_trace(od: Path, name: str, trace: dict) -> Path:
    tdir = od / "traces"
    tdir.mkdir(parents=True, exist_ok=True)
    path = tdir / f"{name}.json"
    path.write_text(json.dumps(trace, indent=2, ensure_ascii=False) + "\n",
                    encoding="utf-8")
    return path


def _trace_stats(od: Path) -> dict:
    files = sorted((od / "traces").glob("*.json"))
    counted = passes = revisions = 0
    for f in files:
        try:
            tr = json.loads(f.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        counted += 1
        if tr.get("verifier", {}).get("verdict") == "pass":
            passes += 1
        revisions += int(tr.get("revisions", 0))
    return {"traces": len(files),
            "verifier_pass_rate": round(passes / counted, 2) if counted else 0.0,
            "revisions": revisions}


def _run_loop(task: dict) -> dict:
    used_before = roles.llm_used_count()
    plan = roles.plan(task)
    evidence = roles.investigate(plan)
    drafted = roles.draft(task, evidence)
    check = verifier.verify(drafted["note"], evidence)
    revisions = 0
    published = drafted["note"]
    published_source = drafted["source"]
    if check["verdict"] == "fail":
        revisions = 1
        revised = roles.revise(task, evidence, drafted["note"],
                               check["mismatches"])
        recheck = verifier.verify(revised["note"], evidence)
        if recheck["verdict"] == "pass":
            drafted, check = revised, recheck
            published = revised["note"]
            published_source = revised["source"]
        else:
            check = recheck
            published = (roles.template_narrative(task, evidence)
                         if task.get("type") == "monthly_review"
                         else roles.template_note(task, evidence))
            published_source = "template_fallback"
    return {
        "task": task,
        "plan": plan,
        "evidence": evidence,
        "draft": drafted["note"],
        "draft_source": drafted["source"],
        "verifier": check,
        "revisions": revisions,
        "published": published,
        "published_source": published_source,
        "llm_used": roles.llm_used_count() > used_before,
        "likely_drivers": drafted["likely_drivers"],
    }


def _update_markdown(md_path: Path, block: dict, trace_ref: str) -> None:
    drivers = "\n".join(f"- {d}" for d in block["likely_drivers"])
    section = (
        "\n## Investigation\n\n"
        "**Likely drivers**\n\n" + drivers + "\n\n"
        "**Note**\n\n" + block["note"] + "\n\n"
        f"_Trace: {trace_ref}_\n"
    )
    text = ""
    if md_path.exists():
        text = md_path.read_text(encoding="utf-8")
        idx = text.find("\n## Investigation")
        if idx >= 0:
            text = text[:idx]
    md_path.write_text(text.rstrip() + "\n" + section, encoding="utf-8")


def run_investigation(date: str, worker_id: str,
                      data_dir=None, out_dir=None) -> dict:
    _dirs(data_dir, out_dir)
    od = Path(out_dir) if out_dir is not None else shiftlens.out_dir()
    report_path = od / "daily_reports" / date / f"{worker_id}.json"
    if not report_path.exists():
        raise FileNotFoundError(
            f"{report_path} missing — run `report` (or `demo`) first; "
            "the agentic pass needs v1 daily reports")
    rep = json.loads(report_path.read_text())
    task = {"type": "investigation", "date": date, "worker_id": worker_id,
            "flags": list(rep.get("flags", []))}

    trace = _run_loop(task)

    block = {"likely_drivers": trace["likely_drivers"],
             "evidence": trace["evidence"],
             "note": trace["published"]}
    rep["investigation"] = block
    report_path.write_text(
        json.dumps(rep, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    trace_name = f"{date}_{worker_id}"
    _update_markdown(report_path.with_suffix(".md"), block,
                     f"out/traces/{trace_name}.json")
    _write_trace(od, trace_name, trace)
    return trace


def run_monthly_review(data_dir=None, out_dir=None) -> dict:
    _dirs(data_dir, out_dir)
    od = Path(out_dir) if out_dir is not None else shiftlens.out_dir()
    monthly_path = od / "monthly.json"
    if not monthly_path.exists():
        raise FileNotFoundError(
            f"{monthly_path} missing — run `monthly` (or `demo`) first; "
            "the agentic pass needs v1 monthly analytics")
    monthly = json.loads(monthly_path.read_text())
    task = {"type": "monthly_review", "period": monthly.get("period")}

    trace = _run_loop(task)
    trace.pop("likely_drivers", None)
    _write_trace(od, "monthly", trace)

    monthly["narrative"] = trace["published"]
    monthly["agentic"] = _trace_stats(od)
    monthly_path.write_text(
        json.dumps(monthly, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8")
    return trace


def format_trace_summary(trace: dict) -> str:
    task = trace["task"]
    label = (f"{task['date']}/{task['worker_id']}"
             if task["type"] == "investigation" else "monthly")
    verdict = trace["verifier"]["verdict"]
    fallback = (" [template fallback]"
                if trace.get("published_source") == "template_fallback"
                else "")
    mode = "llm" if trace.get("llm_used") else "offline"
    return (f"{label}: task={task['type']}, "
            f"tool calls={len(trace['plan'])}, verifier={verdict}{fallback}, "
            f"revisions={trace['revisions']} ({mode})")
