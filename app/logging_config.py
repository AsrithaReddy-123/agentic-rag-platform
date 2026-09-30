"""JSON logs on stdout so a container log driver can ship them."""

from __future__ import annotations

import json
import logging
import sys

_CONFIGURED = False


def configure_logging() -> None:
    global _CONFIGURED
    if _CONFIGURED:
        return
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("%(message)s"))
    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(logging.INFO)
    _CONFIGURED = True


def log_event(event: str, **fields: object) -> None:
    configure_logging()
    logging.getLogger("agentic_rag").info(json.dumps({"event": event, **fields}))
