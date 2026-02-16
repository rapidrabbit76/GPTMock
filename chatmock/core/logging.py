from __future__ import annotations

import json
from typing import Any


def log_json(prefix: str, payload: Any, logger=None) -> None:
    """
    Log a payload as JSON with a prefix.
    
    Args:
        prefix: Label for the log entry
        payload: Data to log (will be JSON-serialized if possible)
        logger: Optional callable for logging (defaults to print)
    """
    if logger is None:
        logger = print
    
    try:
        logger(f"{prefix}\n{json.dumps(payload, indent=2, ensure_ascii=False)}")
    except Exception:
        try:
            logger(f"{prefix}\n{payload}")
        except Exception:
            pass
