# ShiftLens

*By group **AeroApex** (<Egqalhari@gnu.ac.kr>) — AI Agent Competition, October 2026.*

Agentic workforce analytics: daily worker data (sensors + micro-surveys +
object-detection events) → agent-authored daily reports → a monthly dashboard
correlating wellbeing and conditions with productivity.

**The LLM phrases; it never decides.** All metrics, flags, correlations and
findings are deterministic Python; an optional local LLM (LM Studio) only
rewrites prose. New in v2: a **multi-agent layer** — a five-role agent team
that investigates flagged worker-days and drafts the monthly narrative through
a read-only tool registry, with a verifier agent that re-computes every
numeric claim before anything is published. Fully offline-capable.

## Quickstart

```bash
# from this directory, using the shared venv
../.venv/bin/python -m shiftlens.cli demo      # simulate 30 days, write daily
                                               # reports + monthly.json, then run
                                               # the agentic pass (first 3 flagged
                                               # days + monthly narrative) and
                                               # print findings + trace summaries
../.venv/bin/python -m shiftlens.cli serve     # dashboard on http://127.0.0.1:8200
```

See `DESIGN.md` for the full architecture contract, `docs/` for the report,
technical specification, slide deck and video script.

## Usage

```bash
../.venv/bin/python -m shiftlens.cli simulate
../.venv/bin/python -m shiftlens.cli report [--date D]
../.venv/bin/python -m shiftlens.cli monthly
../.venv/bin/python -m shiftlens.cli agents [--date D]
../.venv/bin/python -m shiftlens.cli demo
../.venv/bin/python -m shiftlens.cli demo --all-agents
../.venv/bin/python -m shiftlens.cli serve
```

## Agentic workflow (v2)

The v1 pipeline answers *what happened*. The agentic layer
(`shiftlens/agents/` — roles, tools, orchestrator, verifier) answers *why —
and are we sure?*. A team of role agents works over the same deterministic
data; the LLM may plan and draft, but every number that reaches output is
either computed by a tool or verified against one. With no LLM, the same loop
runs on heuristic plans and template drafts.

### Roster

```
        task: a flagged worker-day, or the monthly review
                           │
                           ▼
                    ┌─────────────┐
                    │ orchestrator│  runs the loop, writes the trace
                    └──────┬──────┘
          ┌───────────┬────┴─────┬────────────┐
          ▼           ▼          ▼            ▼
      ┌────────┐ ┌────────────┐ ┌────────┐ ┌──────────┐
      │planner │ │investigator│ │drafter │ │ verifier │
      │ordered │ │executes    │ │prose   │ │the critic:│
      │tool-   │ │tool calls, │ │from    │ │recomputes│
      │call    │ │collects    │ │evidence│ │every     │
      │plan    │ │evidence    │ │only    │ │number    │
      └───┬────┘ └─────┬──────┘ └───┬────┘ └────┬─────┘
          └────────────┴── tool registry ───────┘
```

For each task, the orchestrator runs **plan → investigate → draft → verify →
(revise once if rejected) → publish**. The verifier re-computes numeric claims
against deterministic tool data before publication.

### Tool registry

- `get_worker_day(date, worker_id)`
- `get_worker_history(worker_id, last_n=7)`
- `compare_to_team(date, metric)`
- `get_environment_context(date)`
- `get_flag_rules()`
- `get_correlations()`
- `get_findings()`
- `get_flagged_days(worker_id)`

### Traces and outputs

Each task writes a JSON trace under `out/traces/` capturing the task, plan,
evidence, draft, verifier verdict, revision count, published text and whether
the LLM was used. Outputs are additive: flagged daily reports gain an
`investigation` block and `out/monthly.json` gains a verified `narrative`
plus agentic statistics.
