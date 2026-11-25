"""LangSmith tracing is for local development only (owner's decision, 25 Nov 2026).

Traces hold users' questions and answers, and the brief says user logs never go to an external API.
So the serving paths (helpdesk.api in Docker, and anything else that serves users) call
serving_guard() before LangChain/LangGraph is imported: it switches tracing off whatever the
environment says. Local runs (helpdesk.cli, eval scripts) trace when LANGSMITH_TRACING=true.
HELPDESK_TRACING=1 lets a local API trace too (a developer's own questions, never a public host).
"""
from __future__ import annotations

import os

TRACE_VARS = ("LANGSMITH_TRACING", "LANGSMITH_TRACING_V2", "LANGCHAIN_TRACING_V2", "LANGCHAIN_TRACING")


def serving_guard() -> bool:
    """Turn tracing off unless HELPDESK_TRACING=1. Returns True when tracing stays allowed."""
    if os.getenv("HELPDESK_TRACING") == "1":
        return True
    for k in TRACE_VARS:
        os.environ[k] = "false"
    return False
