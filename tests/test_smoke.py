"""ShiftLens end-to-end smoke test (DESIGN.md contract).

Runs the real CLI (`python -m shiftlens.cli demo`) in a subprocess with
SHIFTLENS_DATA_DIR / SHIFTLENS_OUT_DIR pointed at a temp dir, then asserts:
  * data/daily has 30 files, workers.json has 8 workers
  * monthly.json exists with exactly 7 correlations (n = 240)
  * planted signals show: fatigue-units r < -0.3 (plus noise/temp/phone)
  * daily reports exist for all 8 workers on day 1 (json + md)
  * summaries list the real numbers; simulation is byte-deterministic
  * v2 agentic layer: traces exist for the investigated days (plan +
    evidence + verifier verdict each), investigated reports gain an
    "investigation" block, monthly.json gains a non-empty "narrative" and
    an "agentic" summary with verifier pass rate >= 0.5, and the standalone
    `agents --date D` command runs on the existing artifacts
  * the verifier is a real critic: invented numbers fail, evidence-backed
    numbers pass

The LLM is pointed at a dead port by default (SHIFTLENS_LMS_HOST) so the
test exercises the offline heuristic/template path deterministically; set
SHIFTLENS_LMS_HOST yourself to test against a live LM Studio.

Works under pytest, or as a plain script:  python tests/test_smoke.py
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def _run_cli(tmp: Path, *argv: str) -> subprocess.CompletedProcess:
    env = dict(os.environ)
    env["SHIFTLENS_DATA_DIR"] = str(tmp / "data")
    env["SHIFTLENS_OUT_DIR"] = str(tmp / "out")
    env.setdefault("SHIFTLENS_LMS_HOST", "http://127.0.0.1:9")
    env.setdefault("SHIFTLENS_LMS_TIMEOUT", "5")
    proc = subprocess.run(
        [sys.executable, "-m", "shiftlens.cli", *argv],
        cwd=REPO, env=env, capture_output=True, text=True, timeout=300,
    )
    assert proc.returncode == 0, (
        f"CLI {argv} exited {proc.returncode}:\n{proc.stdout}\n{proc.stderr}")
    return proc


def test_metric_formulas():
    sys.path.insert(0, str(REPO))
    from shiftlens.agent import metrics as m

    rec = {"survey": {"fatigue": 5}, "sensors": {"hr_avg": 120},
           "detections": {"phone_events": 5, "idle_events": 10,
                          "ppe_ok_pct": 90.0, "safety_zone_violations": 1}}
    assert abs(m.fatigue_index(rec) - 1.0) < 1e-9
    assert abs(m.focus_score(rec) - 0.3) < 1e-9
    assert abs(m.safety_score(rec) - 0.65) < 1e-9
    hi = {"detections": {"phone_events": 50, "idle_events": 50,
                         "ppe_ok_pct": 100.0, "safety_zone_violations": 0}}
    assert m.focus_score(hi) == 0.0
    assert m.safety_score(hi) == 1.0
    assert abs(m.pearson_r([1, 2, 3], [2, 4, 6]) - 1.0) < 1e-9
    assert m.pearson_r([1, 1, 1], [2, 3, 4]) == 0.0
    assert m.productivity_normalized(150, 100) == 1.5


def test_demo_pipeline():
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        proc = _run_cli(tmp, "demo")
        assert "Findings:" in proc.stdout

        daily = sorted((tmp / "data" / "daily").glob("*.json"))
        assert len(daily) == 30, f"expected 30 daily files, got {len(daily)}"
        workers = json.loads((tmp / "data" / "workers.json").read_text())["workers"]
        assert len(workers) == 8
        assert all(0.85 <= w["base_skill"] <= 1.20 for w in workers)
        first = json.loads(daily[0].read_text())
        assert first["date"] == "2026-09-01"
        assert len(first["workers"]) == 8
        assert {"date", "environment", "workers"} == set(first)

        monthly = json.loads((tmp / "out" / "monthly.json").read_text())
        assert monthly["period"] == "2026-09"
        assert len(monthly["team_daily"]) == 30
        assert len(monthly["workers"]) == 8
        corrs = {(c["x"], c["y"]): c for c in monthly["correlations"]}
        assert len(monthly["correlations"]) == 7
        assert corrs[("fatigue", "units")]["n"] == 240
        assert corrs[("fatigue", "units")]["r"] < -0.3
        assert corrs[("noise", "defects")]["r"] > 0.2
        assert corrs[("temperature", "ppe_ok_pct")]["r"] < -0.2
        assert corrs[("phone_events", "units")]["r"] < 0
        assert all(c["insight"] for c in monthly["correlations"])
        assert 3 <= len(monthly["findings"]) <= 5

        day1 = tmp / "out" / "daily_reports" / "2026-09-01"
        json_reports = sorted(day1.glob("*.json"))
        assert len(json_reports) == 8
        assert len(list(day1.glob("*.md"))) == 8
        rep = json.loads(json_reports[0].read_text())
        assert {"worker_id", "date", "metrics", "flags", "summary"} <= set(rep)
        assert {"fatigue_index", "focus_score", "safety_score",
                "units", "defects"} == set(rep["metrics"])
        assert str(rep["metrics"]["units"]) in rep["summary"]

        tmp2 = tmp / "again"
        _run_cli(tmp2, "simulate")
        for name in ("workers.json", "daily/2026-09-01.json",
                     "daily/2026-09-15.json", "daily/2026-09-30.json"):
            assert ((tmp / "data" / name).read_bytes()
                    == (tmp2 / "data" / name).read_bytes()), name


def test_verifier_critic():
    sys.path.insert(0, str(REPO))
    from shiftlens.agents import verifier

    evidence = [
        {"tool": "get_worker_day",
         "args": {"date": "2026-09-01", "worker_id": "W01"},
         "result": {"date": "2026-09-01", "worker_id": "W01",
                    "metrics": {"units": 142, "defects": 2,
                                "fatigue_index": 0.643}}},
    ]
    ok = verifier.verify(
        "On 2026-09-01, W01 produced 142 units with a fatigue index of 0.64.",
        evidence)
    assert ok["verdict"] == "pass", ok
    bad = verifier.verify("W01 produced 999 units.", evidence)
    assert bad["verdict"] == "fail", bad
    assert bad["mismatches"] and bad["mismatches"][0]["claim"] == "999"
    assert ok["claims"] == 5, ok
    empty = verifier.verify("No numeric claims here.", evidence)
    assert empty["verdict"] == "pass" and empty["claims"] == 0


def test_agentic_layer():
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        proc = _run_cli(tmp, "demo")
        assert "Agentic pass" in proc.stdout
        out = tmp / "out"

        traces_dir = out / "traces"
        trace_files = sorted(traces_dir.glob("*.json"))
        assert len(trace_files) >= 4, (
            f"expected >= 4 traces (3 investigations + monthly), "
            f"got {len(trace_files)}")
        assert (traces_dir / "monthly.json").exists()

        for tp in trace_files:
            tr = json.loads(tp.read_text())
            assert {"task", "plan", "evidence", "draft", "verifier",
                    "revisions", "published", "llm_used"} <= set(tr), tp.name
            assert tr["plan"], tp.name
            assert tr["evidence"], tp.name
            assert all({"tool", "args"} <= set(e) for e in tr["evidence"])
            assert tr["verifier"]["verdict"] in ("pass", "fail"), tp.name
            assert isinstance(tr["revisions"], int)
            assert tr["published"].strip()

        monthly = json.loads((out / "monthly.json").read_text())
        assert len(monthly["correlations"]) == 7
        assert monthly["narrative"].strip()
        ag = monthly["agentic"]
        assert ag["traces"] == len(trace_files)
        assert ag["verifier_pass_rate"] >= 0.5, ag
        assert ag["revisions"] >= 0

        investigated = []
        for p in sorted((out / "daily_reports").glob("*/*.json")):
            rep = json.loads(p.read_text())
            if "investigation" not in rep:
                continue
            blk = rep["investigation"]
            assert {"likely_drivers", "evidence", "note"} <= set(blk), p
            assert blk["note"].strip() and blk["likely_drivers"], p
            assert {"worker_id", "date", "metrics", "flags",
                    "summary"} <= set(rep), p
            assert "## Investigation" in p.with_suffix(".md").read_text(), p
            investigated.append(p)
        assert len(investigated) >= 3, (
            f"expected >= 3 investigated reports, got {len(investigated)}")

        proc2 = _run_cli(tmp, "agents", "--date", "2026-09-01")
        assert "Agentic pass" in proc2.stdout
        monthly2 = json.loads((out / "monthly.json").read_text())
        assert monthly2["narrative"].strip()
        assert monthly2["agentic"]["traces"] == len(
            list(traces_dir.glob("*.json")))
        assert monthly2["agentic"]["verifier_pass_rate"] >= 0.5


if __name__ == "__main__":
    test_metric_formulas()
    print("ok - metric formulas")
    test_demo_pipeline()
    print("ok - demo pipeline (30 days, 8 workers, 7 correlations, reports)")
    test_verifier_critic()
    print("ok - verifier critic (numeric claims re-checked vs tool data)")
    test_agentic_layer()
    print("ok - agentic layer (traces, investigations, monthly narrative)")
    print("SMOKE TEST PASSED")
