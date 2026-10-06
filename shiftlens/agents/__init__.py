"""ShiftLens v2 agentic layer (DESIGN.md contract).

A team of role agents answers "why — and are we sure?" over the same
deterministic v1 data:

- agents/tools.py        — the tool registry; the ONLY way agents read data
- agents/roles.py        — planner / investigator / drafter role agents
- agents/verifier.py     — the critic: re-checks every numeric claim in a
                           draft against tool data before anything publishes
- agents/orchestrator.py — runs plan -> investigate -> draft -> verify ->
                           (revise once) -> publish, and writes traces

The LLM still never decides: with LM Studio offline the same loop runs on
heuristic plans and template drafts.
"""
from __future__ import annotations
